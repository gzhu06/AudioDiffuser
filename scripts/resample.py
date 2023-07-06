import glob, os, shutil
import subprocess

SR = 24000

def resample_file(filelist, outfolder):
    """
    downsample musdb18 stem files into separate tracks 

    Args:
        fileList (list): a list of mp4 stem files
        outFolder (string): target folder to store separate audiofile tracks
    """

    # first downsampled to SR
    for input_audiofile in filelist:
        
        output_audiofolder = os.path.join(outfolder, input_audiofile.split('/')[-2])
        os.makedirs(output_audiofolder, exist_ok=True)

        output_audiofile = os.path.join(output_audiofolder, input_audiofile.split('/')[-1])

        cmd = ['ffmpeg', '-i', input_audiofile, '-ac', '1', '-af', 'aresample=resampler=soxr', '-ar', str(SR), output_audiofile]
        completed_process = subprocess.run(cmd)
        
        # confrim process completed successfully
        assert completed_process.returncode == 0
        

def main():
    
    src_wavfolder = '/storageNVME/ge/youtube_dataset/youtube_dataset_mono_remov_silence/Piano'
    tar_wavfolder = '/storageNVME/ge/youtube_dataset/youtube_dataset_mono_remov_silence_24k'
    filelist = glob.glob(os.path.join(src_wavfolder, '**/*.wav'), recursive=True)
    print(len(filelist))
    resample_file(filelist, tar_wavfolder)
    
if __name__ == "__main__":
    main()