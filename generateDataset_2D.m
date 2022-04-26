nSample = 10000;    % # of samples, changing this number is OKAY. 

shape_size = 32; % # of pixels for a shape, each shape is an image
encoderDepth = 2; % the encoderDepth for unet, 


%%%%%%%%%%%%%%% simulation parameters %%%%%%%%%%%%%%%%
energy_half_width = 0.5;

nSub = 2^encoderDepth;   % the distorted image has shape_size*nSub = 32*4 = 128 pixels

blur_max = 1;   % number of macro pixel
noise_amp_max = 0.5;
noise_range = 2;
K_gauss_max = 2;  % shape max smoothness

save_file = true;

rng(1); % fix random seed
K_gauss = K_gauss_max*rand(1,1);
p = 0.5;
margin = ceil(K_gauss)+2;
K_fuse = energy_half_width*nSub;

%%%%%%%%%%%%%%% make folder %%%%%%%%%%%%%%%%
mkdir('dataset')
mkdir('dataset/distorted')
mkdir('dataset/original')


for ii = 1:nSample
    %%%%%%%%%%%%%%% original %%%%%%%%%%%%%%%%
    firstLayer=randi(2,shape_size,shape_size)-1;
    cc=zeros(shape_size+margin*2,shape_size+margin*2);
    cc((margin+1):(margin+shape_size),(margin+1):(margin+shape_size))=firstLayer;
    cc=imgaussfilt(cc,K_gauss);
    firstLayer=cc((margin+1):(margin+shape_size),(margin+1):(margin+shape_size));
    shape=firstLayer>p;

%     CC = bwconncomp(shape,8);
%     numPixels = cellfun(@numel,CC.PixelIdxList);
%     [biggest,idx] = max(numPixels);
%     shape_new = zeros(size(shape));
%     shape_new(CC.PixelIdxList{idx}) = shape(CC.PixelIdxList{idx});
    
    shape_new = shape.*1.0;
    shape_new = (imgaussfilt(shape_new,5)>0.4)*1.0;
    %shape_new = shape_new.*(sqrt(rand(size(shape_new))));


    blur = blur_max*rand(1,1);   % number of macro pixel
    noise_amp = noise_amp_max*rand(1,1);

    shape_new = imgaussfilt(shape_new,blur);

    sdf = bwdist(shape_new==0,'euclidean'); % signed distance function

    shape_with_noise = shape_new.*((1-noise_amp*0.5) + noise_amp*(rand(size(shape_new))));
    shape_with_noise(shape_with_noise>1)=1;
    shape_new(sdf<=noise_range) = shape_with_noise(sdf<=noise_range);

    %%%%%%%%%%%%%%% distortion %%%%%%%%%%%%%%%%
    superResolution = 3;
    shape_sub = imresize(shape_new, nSub*superResolution);
    HeatDefused = imgaussfilt(shape_sub,K_fuse*superResolution);
    
    out = (HeatDefused>0.5)*1.0;
    out = imresize(out, 1/superResolution, 'bilinear');
    
    %%%%%%%%%%%%%%% write image %%%%%%%%%%%%%%%%
    if (save_file)
        imwrite(shape_new,['dataset/original/' int2str(ii) '.png']);
        imwrite(out,['dataset/distorted/' int2str(ii) '.png']);
    end
end
