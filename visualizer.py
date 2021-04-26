import errno
import math
import librosa
import numpy as np
import scipy.signal
import tempfile
import os
import shutil
import multiprocessing
from multiprocessing import Pool
import subprocess
import sys
import argparse
from pathlib import Path

from PIL import Image
from PIL import ImageChops

### Tunable values ###

SR = 22050			# Audio sample rate
BANDS = 9			# How many frequency bands?
FPS = 15			# Video FPS
N_SEGMENTS = 8		# Number of segments of each band
RESOLUTION = 1080	# Video resolution (1080/4K)
KEEP_FRAMES = False

### Helper functions ####

# to find the right settings to avoid drift as possible, thanks to Jon Nordby
# https://stackoverflow.com/a/67077404/8049293
def _next_power_of_2(x):
	"""Return the next power of two given x"""

	return 2**(math.ceil(math.log(x, 2)))

def _params_for_fps(fps=30, sr=16000):
	"""Given a frame rate and sample rate gives back the best parameters for stft"""

	frame_seconds=1.0/fps
	frame_hop = round(frame_seconds*sr) # in samples
	frame_fft = _next_power_of_2(2*frame_hop)
	rel_error = (frame_hop-(frame_seconds*sr))/frame_hop
	
	return frame_hop, frame_fft, rel_error

def tune_parameters(fps=30, sr=22050):
	"""Given some frame rate and sample rate, tries to reduce as much a possible the time drift of the resulting video compared to the original audio
	by slightly modifying the sample rate"""

	# Initialize two utility variables. prev_sr stores the last two sample rates tested,
	# prev_err the last two error rates
	prev_sr = [None, None]
	prev_err = [None, None]
	#Let's try one at a time adjusting while we go, till we reach equilibrium
	while True:
		frame_hop, frame_fft, rel_error = _params_for_fps(fps=fps, sr=sr)
		# If the error is (or gets to) zero we're done
		if rel_error == 0:
			return sr, frame_hop, frame_fft, rel_error
		# Alternatively we try to optimize changing the sample rate by one
		elif rel_error < 0:
			# If we encountered the value already, that means we have reached the limit
			# and we are oscillating between negative and positive error
			if rel_error == prev_err[-2]:
				break 
			# We keep track our last two results with the new one
			prev_sr.append(sr)
			prev_sr.pop(0)
			prev_err.append(rel_error)
			prev_err.pop(0)
			# We are drifting longer than we should, so we try to get better reducing the sample rate
			sr -=1
		elif rel_error > 0:
			if rel_error == prev_err[-2]:
				break
			prev_sr.append(sr)
			prev_sr.pop(0)
			prev_err.append(rel_error)
			prev_err.pop(0)
			sr +=1
		# We reached the limit in which one value is too little and one too much. We pick the least significant error between the two
	# The last value we evaluated was the same as our prev_err[0](prev_err[-2]), so we check if this is the best or the one before was better
	if abs(prev_err[1]) < abs(prev_err[0]):
		sr = prev_sr[1]
	frame_hop, frame_fft, rel_error = _params_for_fps(fps=fps, sr=sr)
	return sr, frame_hop, frame_fft, rel_error

# Most of the sound analysis code is taken directly from the one provided by AKX here:
# https://stackoverflow.com/a/67077332/8049293

def open_audio_file(file_path, sr):
	"""Load an audio file using librosa
	https://librosa.org/doc/latest/generated/librosa.load.html
	"""
	y, sr = librosa.load(file_path, sr=sr)
	return y,sr


def generate_spectrogram_matrix (fps=10, frame_fft=2048, frame_hop=512, bands=9, n_segments=8):
	"""Generate a spectrogram matrix for the input audio using librosa short time fourier transform.
	Parameters
	----------
	fps : int
		desired framerate for output video. One image will be generated for every frame.
	frame_fft : int
		length of the windowed signal after padding with zeros, see librosa doc for more info.
	frame_hop : int
		number of audio samples between adjacent STFT columns, see librosa doc for more info.
	bands : int
		number of frequency bands in which the frequency spectrum will be divided.
	n_segments : int
		number of segments for every frequency band.
		
	Returns
	-------
	sp_matrix : numpy array
    	a 2D matrix of shape (#bands,#frames)
	"""
	hop_length_secs = 1/fps
	stft = np.abs(librosa.stft(y, hop_length=frame_hop, n_fft=frame_fft))
	
	num_bins, num_samples = stft.shape

	# Resample to the desired number of frequency bins
	stft2 = np.abs(scipy.signal.resample(stft, bands, axis=0))
	stft2 = stft2 / np.max(stft2)  # Normalize to 0..1

	# Remap the 0..1 signal to integer indices
	# -- the square root boosts the otherwise lower signals for better visibility.
	sp_matrix = (np.sqrt(stft2) * n_segments).astype(np.uint8)
	# TOTRY
	# -- NON square root boosts the otherwise lower signals for better visibility.
	#sp_matrix = (stft2 * n_segments).astype(np.uint8)
	return sp_matrix

def _generate_frames(spectrogram_matrix, frame_index, n_frames, image_folder, tmp_folder, keep_frames=False):
	"""Internal function to generate frame images. Needed for multiprocess. Should only be called through the main generate_frames function.
	frame_index is an array containing the indices for the frames in the spectrogram_matrix. Needed to split the frame generation in multiple processes.
	"""
	sm = spectrogram_matrix
	nchar = len(f"{n_frames}")
	this_thread_frames = sm.shape[1]
	thread_id = os.getpid()
	# Open background
	# (I suppose this might be made more efficient not loading the image for every thread. but I have no idea on how to share data between threads)
	try:
		background = Image.open(f"{image_folder}/base.jpg") #.convert('RGB')
	except FileNotFoundError:
		print("No base file found. Base file must in the image folder and be called 'base.jpg'")
		raise FileNotFoundError
	# Load all the bars images into dictionary:
	# Same as background. If you know how to do it, feel free to optimize.
	imms = {}
	for i in range(BANDS):
		for j in range(N_SEGMENTS):
			col = i+1
			row = j+1
			key = f"{col}_{row}"
			try:
				imms[key] = Image.open(f"{image_folder}/{col}_{row}.png")
			except FileNotFoundError:
				print(f"File '{col}_{row}.png' not found. Files for the bars and segments must be in the image folder, must be named following the format col_row.png', and must be a png file with transparency")
				raise FileNotFoundError
	# Loop through matrix and generate the frames
	# i is the frame #
	for i in range(sm.shape[1]):
		bg = ImageChops.duplicate(background)
		# j is the current bar
		for j in range(sm.shape[0]):
			# With slicing we get how many segments are active in that bar at that frame
			segments = sm[j,i]
			# If there are zero segments we don't need to paste any image
			if segments:
				fg = imms[f"{j+1}_{segments}"]
				bg.paste(fg, mask=fg)
		num = f"{frame_index[i]+1}".zfill(nchar)
		# Save the file
		filename = f"{tmp_folder}/{num}.jpg"
		bg.save(filename)
		# I suppose there's a way to aggregate the threads work and print a general progress indicator, or even better use tqdm.
		print(f"thread {thread_id} generated frame {i+1} of {this_thread_frames}")

def generate_frames(spectrogram_matrix, image_folder, tmp_folder, keep_frames=KEEP_FRAMES):
	"""Function to generate frames from a spectrogram matrix.
	Parameters
	----------
	spectrogram_matrix : numpy array
		a 2d numpy array of shape (#bands,#frames) containing the visualization informations
	image_folder : string
		the source folder of the images that will be assembled
	tmp_folder : string
		the temporary folder where images will be saved
	keep_frames : bool
		If true the images of the frames will be kept, if false they will be deleted after assembling the video
	"""
	n_frames = spectrogram_matrix.shape[1]
	# Get the number of available threads:
	thread_n = multiprocessing.cpu_count()
	# split spectrogram matrix in N parts (N=#threads)
	splitted_matrices = np.array_split(spectrogram_matrix, thread_n, axis=1)
	# Create an index array for each section
	index = np.array(range(spectrogram_matrix.shape[1]))
	splitted_indices = np.array_split(index, thread_n)
	# Create a pool of tasks with multiprocessing, something that I really don't fully understand
	p = Pool(thread_n)
	# To pass different arguments (because of the different frames and stuff) here we construct a list of these parameters
	# for each process to elaborate. Except for matrix and indices they stay the same.
	#I suppose there is a smarter way to do this.
	arguments = [[partial_matrix, partial_index, n_frames, image_folder, tmp_folder, keep_frames] for partial_matrix, partial_index in zip(splitted_matrices, splitted_indices)]
	# Starmap is like map but it accepts multiple arguments
	p.starmap(_generate_frames, arguments)
	p.terminate()
	# To keep the frames as single images we copy the content of the temporary folder to a /frame/ folder where the script is.
	if KEEP_FRAMES:
		if os.path.exists("./frames/"):
			shutil.rmtree("./frames/")
		shutil.copytree(tmp_folder, "./frames/")

def generate_video():
	"""Generate a video file from a bunch of image files. It relies on ffmpeg to do the job.
	For this reason the supported formats etc are the ones supported by ffmpeg."""
	
	n_frames = len(os.listdir(tmp_dir))
	n_char = str(len(f"{n_frames}")).zfill(2)
	if os.path.isfile("./temp_output.mp4"):
		os.remove("./temp_output.mp4")
	subprocess.run(['ffmpeg', '-framerate', f'{FPS}', '-i', f'{tmp_dir}/%{n_char}d.jpg', './temp_output.mp4'], stderr=sys.stderr, stdout=sys.stdout)
	subprocess.run(['ffmpeg','-i', './temp_output.mp4', '-i', f'{audio_file_path}', '-c:v', 'copy', 'output.mp4'], stderr=sys.stderr, stdout=sys.stdout)
	os.remove("./temp_output.mp4")



if __name__=="__main__":
	parser = argparse.ArgumentParser(description='A little python script that generates a video visualization for audio files. With wooden blocks!')
	parser.add_argument("file_path", type=Path,
			help='the audio file you want to generate video from')
	parser.add_argument('--fps', nargs='?',
			help='the fps of the output file. One image will be created for every frame. Default is 15.')
	parser.add_argument('--resolution', nargs='?', type=str, choices=['1080', '4K'],
			help='the resolution of the output file. Can choose between 1080 and 4K. Default is 1080.')
	parser.add_argument('--keep_frames', action='store_true',
			help='Preserve all the frames generated to make the video. By default they are removed after the video is been created.')
	p = parser.parse_args()

	#For convenience the same defaults are on top of the file, we check if they are provided by the user and overwrite if needed
	if p.fps:
		FPS = p.fps
	if p.resolution:
		RESOLUTION = p.resolution
	if p.keep_frames:
		KEEP_FRAMES = True

	
	audio_file_path = p.file_path
	image_folder = f"./assets/img/{RESOLUTION}/"
	
	print ("\nTuning parameters\n")
	sr, frame_hop, frame_fft, rel_error = tune_parameters(FPS, SR)
	print ("\nOpening file...\n")
	y, sr = open_audio_file(audio_file_path, sr)
	duration = librosa.get_duration(y=y, sr=sr)
	drift = rel_error * duration
	print(f"\nYour video will drift {drift*FPS:.2f} frames({drift:.2f} seconds) realtive to the {duration} seconds audio\n")
	spectrogram_matrix = generate_spectrogram_matrix(fps=FPS, frame_fft=frame_fft, frame_hop=frame_hop, bands=BANDS,  n_segments=N_SEGMENTS)
	#Make a temporary folder
	with tempfile.TemporaryDirectory() as tmp_dir:
		generate_frames(spectrogram_matrix, image_folder, tmp_dir, keep_frames=False)
		generate_video()
		