%% Reshape blurry samples to be the same size as actual sample and make black and white instead of greyscale

mkdir('dataset/reshapeBW'); %create a new directory
filePattern = fullfile('.', 'dataset', 'original','*.png'); %used to count how many samples were created. use this line for PC C:\Users\anguye22\Downloads\dataset\original', 
% for MAC: (('C:','Users','anguye22','Downloads','dataset', 'original'), '*.png');
theFiles = natsortfiles(dir(filePattern));
theFiles = theFiles(~[theFiles.isdir]);

% for loop for each of the samples to resize and make black and white
% instead of greyscale
for k = 1:length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    %fprintf(1, 'Now reading %s\n', fullFileName);
    img = imread(fullFileName);
    resize = imresize(img, 4.0); %resize blurred images to be the same size as actual image (30x30 --> 240x240)
    
    %BW = im2bw(resize, 0.5); %make images bw instead of greyscale with 0.5 threshold
    imwrite(resize,['dataset/reshapeBW/' int2str(k) '.png']); %save new images into new directory
end
