# <center>Using Machine Learning and Stereo Vision to Predict the Position of an Object in an Image</center>

![Poster](media/Research%20Poster%20Template%202024.png)

## <center>Research Question</center>
<center>
Is there a significant difference between the distances predicted by a stereo vision model compared to other machine learning models and actual distances?
</center>

## <center>Background</center>
As artificial intelligence develops increasingly rapidly, there are an ever so increasing number of applications in more and more fields across multiple disciplines. One such subset of artificial intelligence focuses on the understanding and extraction of information from digital images. This subset is called computer vision. 

A type of computer vision called stereo vision utilizes input from 2 perspectives (from pairs of either camera inputs or photos) in order to estimate the depth of different points on an image. The output is a disparity map, which measures the disparity between 2 corresponding points on the 2 input visuals. This can either be a live or still disparity map, depending on whether it is reading live information from 2 cameras or calculating disparity based on 2 photos. A common method to match corresponding locations from 2 inputs is called Semi-Global Block Matching (SGBM), which breaks down images into blocks, then compares different blocks to find matching “locations” on the 2 images. Based on the difference in position relative to the entire image of each block, in combination with the camera calibration data, stereo vision algorithms calculate and output a disparity map containing the relative depth of different points on the image. More advanced algorithms translate the disparity map into a depth map, which calculates actual distances from the camera rather than relative position compared to other points in the image. 

## <center>Tools</center>
- OpenCV: Python image processing library, also contains SGBM sorting algorithm
- GLPN NYU: a depth-estimation model we compared our stereo vision model with
- DPT Hybrid Midas: another depth-estimation model we compared our stereo vision model with
- Logitech C920 HD Pro Webcam: used to capture photos for stereo vision
- calibdb.net: aided in camera calibration
- ChatGPT: aided us with coding the stereo vision model

## <center>Visualizations</center>
we have a bunch of figures

## <center>Methods</center>
- Coded stereo vision model, used SGBM to match images, then took test photos
- Fine-tuned settings for more accurate photos
    - Eliminate noise
    - “Replicating” accurate scene in depth maps
- Imported 2 other depth estimation AI models (non stereo vision)
- Subtracted 3 models’ measurements from real distances
- Performed T-test

## <center>Results</center>
No significant difference between the real distances and the models’ measurements

### P-values

| GLPN-NYU | Intel | Stereo Vision |
|----------|-------|---------------|
|  30.3% | 24.7% | 12.5% | 

## <center>Conclusion</center>
Due to the fact that our study showed that there was no significant difference between the 3 comp. vis. models and the real distances, we can reasonably conclude that the real solution, which in our case is the accurate/real measurements of depth, are contained in the 3 models we tested. However, it is important to note that when simply looking at the outputted depth maps, the GLPN-NYU model had the smoothest graph, while our stereo vision model had the most noise. This is something that should be addressed later before being applied to real-world applications. 

## <center>Code Overview</center>
Folders in the repository:

- `calibrations`: contains camera calibration data for different resolutions
- `data-analysis`: contains the code for comparing the model distances to the real distances (also the T-test)
- `models`: contains the code for the machine learning models we used
- `output`: contains the machine learning outputted depth maps
- `pictures`: contains the images we used to test our setup

Important Files:
TODO

## <center>References</center>
What is Computer Vision?: Microsoft Azure. What Is Computer Vision? | Microsoft Azure. (n.d.). https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-is-computer-vision#object-classification 

Hamzah, R. A., & Ibrahim, H. (2015, December 28). Literature survey on stereo vision disparity map algorithms. Journal of Sensors. https://doi.org/10.1155/2016/8742920 

Ashtari, H. (2022, May 13). What Is Computer Vision? Meaning, Examples, and Applications in 2022. Spiceworks. https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-computer-vision/ 
