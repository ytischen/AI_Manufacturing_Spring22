%% evaluate_prediction.m
% Names: Alisa Nguyen and Jacob Frabutt
% Date: 12/03/2021
% Description: This program trains a neural network (specifically a unet),
%   using data from generateDataset_2D.m to predict the shrinkage of
%   a set of images.
%   Reference documentation for a unet can be found at this page:
%   https://www.mathworks.com/help/vision/ref/unetlayers.html
% NOTE: This program should only be run after generateDataset_2D is run.

%% Data Structuring

% The data files should be stored in a dataset directory located in 
% your current directory
dataSetDir = fullfile('.', 'dataset');
imageDir = fullfile(dataSetDir, 'distorted'); % original images
labelDir = fullfile(dataSetDir, 'original'); % distorted images

% Create an imageDatastore object to store the images
imds = imageDatastore(imageDir);
 
% Determine the number of samples
numSamples = numpartitions(imds);       
 
% Determine Indices for training, testing, and validation:
% Use 70% of images for training
numTrain = round(numSamples*0.7);
trainInd = 1:numTrain;
 
% Use 15% of images for validation
numVal = round(numSamples*0.15);
valInd = (numTrain+1:numTrain+numVal);
 
% Use 15% of images for testing
numTest = round(numSamples*0.15);
testInd = (numTrain+numVal+1:numSamples);
 
% Get the images for each stage
trainingImages = imds.Files(trainInd);
valImages = imds.Files(valInd);
testImages = imds.Files(testInd);

% Create image datastores for training, validation, and testing
imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

imdsTrain = transform(imdsTrain,@(x) rescale(x));
imdsVal = transform(imdsVal,@(x) rescale(x));
imdsTest = transform(imdsTest,@(x) rescale(x));

% % Define the classes and labels of our data
% % Pixels are either white or black
% classNames = ["shapes", "background"];
% labelIDs = [1 0];
 
% Create a pixelLabelDatastore to store the ground truth pixel labels
pxds = imageDatastore(labelDir);%pixelLabelDatastore(labelDir, classNames, labelIDs); 

% Separate the files for each stage
trainingLabels = pxds.Files(trainInd);
valLabels = pxds.Files(valInd);
testLabels = pxds.Files(testInd);

% Create pixel label datastores for training, validation, and testing
% pxdsTrain = pixelLabelDatastore(trainingLabels, classNames, labelIDs);
% pxdsVal = pixelLabelDatastore(valLabels, classNames, labelIDs);
% pxdsTest = pixelLabelDatastore(testLabels, classNames, labelIDs);

pxdsTrain = imageDatastore(trainingLabels,'ReadFcn', @(x)rescale(imread(x)));
pxdsVal = imageDatastore(valLabels,'ReadFcn', @(x)rescale(imread(x)));
pxdsTest = imageDatastore(testLabels,'ReadFcn', @(x)rescale(imread(x)));

% Create a datastore for training and validating the network
dsTrain = combine(imdsTrain, pxdsTrain);
dsVal = combine(imdsVal, pxdsVal);
         

%% Visualize the results

% The following code block alows you to viusalize what the model is doing
% by demonstrating the shrinkage the network predicts. To run this block of
% code, you must have a sample image and its corresponding distorted image
% in the working directory. Here '1.png' is the original image and '1d.png'
% is the corresponding distorted image.
% However, this section can all be commented out if you would like.

% load in the images
I = imread('./dataset/distorted/2.png');
YPred_large = predict(net,I);
YPred = imresize(YPred_large, 1/4, 'bilinear');
YTrue = rescale(imread('./dataset/original/2.png'));

difference = abs(YTrue-YPred);

% overlay = [];
% overlay(:,:,1) = YTrue;
% overlay(:,:,2) = YPred;
% overlay(:,:,3) = 0;

% create an overlay and montage
%   The top left image will be the original image
%   The top right image will the predicted
%   The bottom left image will highlight the discrepancy between the
%       predicted shirnkage and the ground truth shrinkage (error)
%   The bottom right image is the ground truth shrinkage

figure;montage({I,YPred,difference,YTrue})

%% Network Statistics

% Create an label store of the ground truth test images
pxdsTruth = imageDatastore(testLabels,'ReadFcn', @(x)rescale(imread(x)));
% Run the testing images through the network
pxdsResults = predict(net,imdsTest);
pxdsResults = imresize(pxdsResults,1/4,'bicubic');
pxdsResults = double(pxdsResults);

% Load all the ground truth value
groundTruths = zeros(size(pxdsResults));
for ii = 1:length(pxdsTruth.Files)
    groundTruths(:,:,1,ii) = mat2gray(imread(pxdsTruth.Files{ii}));
end

% compare the ground truth and predicted result of 1.png
figure;montage({groundTruths(:,:,1),pxdsResults(:,:,1)})

% find the relationship between predicted values and the truth
% they should be identical
% Ideally, the figure should be a straight line, as y = x
mdl = fitlm(groundTruths(:),pxdsResults(:))
figure;
plot(mdl);
xlabel('True')
ylabel('Predicted');
title('Predicted v.s. Truth');

% compuate the mean squared error among all test images
err = immse(groundTruths, pxdsResults);
fprintf('\n The mean-squared error is %0.4f\n', err);

% plot all the pixel values of predicted and truth
figure;
plot(1:length(groundTruths(:)),groundTruths(:),'.');
hold on;
plot(1:length(groundTruths(:)),pxdsResults(:),'.');

%% Save the network

% the command saves the trained network to the current working directory
save('net.mat', 'net');