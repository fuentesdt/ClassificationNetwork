clear all
close all
close(findall(groot, "Type", "figure"));
%% Create Simple Deep Learning Neural Network for Classification
% This example shows how to create and train a simple convolutional neural network 
% for deep learning classification. Convolutional neural networks are essential 
% tools for deep learning, and are especially suited for image recognition.
% 
% The example demonstrates how to:
%% 
% * Load and explore image data.
% * Define the neural network architecture.
% * Specify training options.
% * Train the neural network.
% * Predict the labels of new data and calculate the classification accuracy.
%% 
% For an example showing how to interactively create and train a simple image 
% classification neural network, see <docid:nnet_gs#mw_a1e3fba3-0eb8-43c7-ae9d-d3e167943fee 
% Create Simple Image Classification Network Using Deep Network Designer>.
%% Load and Explore Image Data
% Load the digit sample data as an image datastore. |imageDatastore| automatically 
% labels the images based on folder names and stores the data as an |ImageDatastore| 
% object. An image datastore enables you to store large image data, including 
% data that does not fit in memory, and efficiently read batches of images during 
% training of a convolutional neural network.
batchsize= 4;
imds = imageDatastore('adcnetwork', 'IncludeSubfolders',true,'ReadSize',batchsize,'LabelSource','foldernames','FileExtensions','.nii','ReadFcn',@mycustomreader);
%% 
% Display some of the images in the datastore.

figure;
perm = randperm(length(imds.Files),20);
for i = 1:20
    subplot(4,5,i);
    xxx = readimage(imds,i);
    imshow(xxx(:,:,64,1 ));
end
%% 
% Calculate the number of images in each category. |labelCount| is a table that 
% contains the labels and the number of images having each label. The datastore 
% contains 1000 images for each of the digits 0-9, for a total of 10000 images. 
% You can specify the number of classes in the last fully connected layer of your 
% neural network as the |OutputSize| argument.

labelCount = countEachLabel(imds)
%% 
% You must specify the size of the images in the input layer of the neural network. 
% Check the size of the first image in |digitData|. Each image is 28-by-28-by-1 
% pixels.

img = readimage(imds,1);
numchannel = size(img,3)
%% Specify Training and Validation Sets
% Divide the data into training and validation data sets, so that each category 
% in the training set contains 750 images, and the validation set contains the 
% remaining images from each label. |splitEachLabel| splits the datastore |digitData| 
% into two new datastores, |trainDigitData| and |valDigitData|.

% LOOCV
Nloocv = 5;
cv = cvpartition( imds.Labels,'KFold',Nloocv )
%mycvmat = [ cv.training(1), cv.training(2), cv.training(3), cv.training(4), cv.training(5), cv.training(6), cv.training(7), cv.training(8)] ;
%[imds.Labels =='1', cv.training(1), cv.training(2), cv.training(3), cv.training(4), cv.training(5), cv.training(6), cv.training(7), cv.training(8), sum(mycvmat,2)] 

pocketchannels = 8;
% Weight minority class by the imbalance ratio (majority count / minority count)
counts = labelCount.Count;
classratio = max(counts) / min(counts);
hyperweight = [classratio];
hyperepoch  = [64];
accuracy = zeros(length(hyperweight ),length(hyperepoch)  );

for idweight =1:length(hyperweight )
for idepoch =1:length(hyperepoch )

%% layers = [
%%     image3dInputLayer([128 128 128 ])
%%     
%%     convolution3dLayer(128,pocketchannels,'Stride',128)
%%     batchNormalizationLayer
%%     reluLayer
%%     % TODO - concatenationLayer
%%     
%%     fullyConnectedLayer(2)
%%     softmaxLayer
%%     classificationLayer('ClassWeights',[1 hyperweight(idweight)],'Classes',unique(imds.Labels))];
layers = [
    image3dInputLayer([256 256 64 ])
    
    convolution3dLayer(3,pocketchannels,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling3dLayer(4,'Stride',4)

    convolution3dLayer(3,pocketchannels,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling3dLayer(4,'Stride',4)

    convolution3dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    globalAveragePooling3dLayer
    dropoutLayer(0.5)
    scalingLayer(Scale=0.25,Offset=0);
    softmaxLayer
    classificationLayer('ClassWeights',[1 hyperweight(idweight)],'Classes',unique(imds.Labels))];
%% %% 
%% % *Image Input Layer* An <docid:nnet_ref.mw_fcd2d9b1-ce25-49d1-9d06-b7cf41594ff4 
% imageInputLayer> is where you specify the image size, which, in this case, is 
% 28-by-28-by-1. These numbers correspond to the height, width, and the channel 
% size. The digit data consists of grayscale images, so the channel size (color 
% channel) is 1. For a color image, the channel size is 3, corresponding to the 
% RGB values. You do not need to shuffle the data because |trainNetwork|, by default, 
% shuffles the data at the beginning of training. |trainNetwork| can also automatically 
% shuffle the data at the beginning of every epoch during training.
% 
% *Convolutional Layer* In the convolutional layer, the first argument is |filterSize|, 
% which is the height and width of the filters the training function uses while 
% scanning along the images. In this example, the number 3 indicates that the 
% filter size is 3-by-3. You can specify different sizes for the height and width 
% of the filter. The second argument is the number of filters, |numFilters|, which 
% is the number of neurons that connect to the same region of the input. This 
% parameter determines the number of feature maps. Use the |'Padding'| name-value 
% pair to add padding to the input feature map. For a convolutional layer with 
% a default stride of 1, |'same'| padding ensures that the spatial output size 
% is the same as the input size. You can also define the stride and learning rates 
% for this layer using name-value pair arguments of <docid:nnet_ref.mw_2d97b6cd-f8aa-4fad-88d6-d34875484820 
% convolution2dLayer>.
% 
% *Batch Normalization Layer* Batch normalization layers normalize the activations 
% and gradients propagating through a neural network, making neural network training 
% an easier optimization problem. Use batch normalization layers between convolutional 
% layers and nonlinearities, such as ReLU layers, to speed up neural network training 
% and reduce the sensitivity to neural network initialization. Use <docid:nnet_ref.mw_b7913af4-3a40-4020-bb2c-18c946f5eadd 
% batchNormalizationLayer> to create a batch normalization layer.
% 
% *ReLU Layer* The batch normalization layer is followed by a nonlinear activation 
% function. The most common activation function is the rectified linear unit (ReLU). 
% Use <docid:nnet_ref.mw_ca5427bd-5cdc-4a58-ba63-302c257d8222 reluLayer> to create 
% a ReLU layer.
% 
% *Max Pooling Layer* Convolutional layers (with activation functions) are sometimes 
% followed by a down-sampling operation that reduces the spatial size of the feature 
% map and removes redundant spatial information. Down-sampling makes it possible 
% to increase the number of filters in deeper convolutional layers without increasing 
% the required amount of computation per layer. One way of down-sampling is using 
% a max pooling, which you create using <docid:nnet_ref.mw_d2785483-a560-4276-a1c0-daa5f58a1d4b 
% maxPooling2dLayer>. The max pooling layer returns the maximum values of rectangular 
% regions of inputs, specified by the first argument, |poolSize|. In this example, 
% the size of the rectangular region is [2,2]. The |'Stride'| name-value pair 
% argument specifies the step size that the training function takes as it scans 
% along the input.
% 
% *Fully Connected Layer* The convolutional and down-sampling layers are followed 
% by one or more fully connected layers. As its name suggests, a fully connected 
% layer is a layer in which the neurons connect to all the neurons in the preceding 
% layer. This layer combines all the features learned by the previous layers across 
% the image to identify the larger patterns. The last fully connected layer combines 
% the features to classify the images. Therefore, the |OutputSize| parameter in 
% the last fully connected layer is equal to the number of classes in the target 
% data. In this example, the output size is 10, corresponding to the 10 classes. 
% Use <docid:nnet_ref.mw_1e7fbc56-4746-4f30-8cd9-7048ce806a0d fullyConnectedLayer> 
% to create a fully connected layer.
% 
% *Softmax Layer* The softmax activation function normalizes the output of the 
% fully connected layer. The output of the softmax layer consists of positive 
% numbers that sum to one, which can then be used as classification probabilities 
% by the classification layer. Create a softmax layer using the <docid:nnet_ref.mw_a09d3c68-d062-4692-a950-9a7fea5c40c3 
% softmaxLayer> function after the last fully connected layer.
% 
% *Classification Layer* The final layer is the classification layer. This layer 
% uses the probabilities returned by the softmax activation function for each 
% input to assign the input to one of the mutually exclusive classes and compute 
% the loss. To create a classification layer, use <docid:nnet_ref.bu5lho8 classificationLayer>.
%% Specify Training Options
% After defining the neural network structure, specify the training options. 
% Train the neural network using stochastic gradient descent with momentum (SGDM) 
% with an initial learning rate of 0.01. Set the maximum number of epochs to 4. 
% An epoch is a full training cycle on the entire training data set. Monitor the 
% neural network accuracy during training by specifying validation data and validation 
% frequency. Shuffle the data every epoch. The software trains the neural network 
% on the training data and calculates the accuracy on the validation data at regular 
% intervals during training. The validation data is not used to update the neural 
% network weights. Turn on the training progress plot, and turn off the command 
% window output.

% TODO hack to deep copy data structures
[imdsTrain,imdsValidation] = splitEachLabel(imds,1,'randomize');
%% Define Neural Network Architecture
% Define the convolutional neural network architecture.
YPred = categorical(NaN(length(imds.Labels),1));
myactivationsone = zeros(length(imds.Labels),1);
myactivationstwo = zeros(length(imds.Labels),1);

global foldmaxaccuracy;
for iii =1:Nloocv
%for iii =1:2
disp(iii);
imdsTrain.Files = imds.Files(cv.training(iii));
imdsTrain.Labels = imds.Labels(cv.training(iii));

imdsValidation.Files = imds.Files(cv.test(iii));
imdsValidation.Labels = imds.Labels(cv.test(iii));
options = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',hyperepoch(idepoch), ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',1, ...
    'Verbose',false, ...
    'MiniBatchSize',batchsize, ...
    'L2Regularization', 1e-4, ...
    'OutputNetwork','best-validation', ...
    'Plots','training-progress',...
    'OutputFcn',@(info)stopTraining(info ));

%% Train Neural Network Using Training Data
% Train the neural network using the architecture defined by |layers|, the training 
% data, and the training options. By default, |trainNetwork| uses a GPU if one 
% is available, otherwise, it uses a CPU. Training on a GPU requires Parallel 
% Computing Toolbox™ and a supported GPU device. For information on supported 
% devices, see <docid:distcomp_ug#mw_57e04559-0b60-42d5-ad55-e77ec5f5865f GPU 
% Support by Release>. You can also specify the execution environment by using 
% the |'ExecutionEnvironment'| name-value pair argument of |trainingOptions|.
% 
% The training progress plot shows the mini-batch loss and accuracy and the 
% validation loss and accuracy. For more information on the training progress 
% plot, see <docid:nnet_ug.mw_507458b6-14c3-4a31-884c-9f2119ff7e05 Monitor Deep 
% Learning Training Progress>. The loss is the cross-entropy loss. The accuracy 
% is the percentage of images that the neural network classifies correctly.

% analyzeNetwork(net{iii})
foldmaxaccuracy = -inf; % reset global
[net{iii}, info{iii}] = trainNetwork(imdsTrain,layers,options);
disp(sprintf('max val accuracy %f',max(info{iii}.ValidationAccuracy)))
%% Classify Validation Images and Compute Accuracy
% Predict the labels of the validation data using the trained neural network, 
% and calculate the final validation accuracy. Accuracy is the fraction of labels 
% that the neural network predicts correctly. In this case, more than 99% of the 
% predicted labels match the true labels of the validation set.

YPred(cv.test(iii)) = classify(net{iii},imdsValidation);
%[predictions scores]  = classify(net{iii},imdsValidation);
layername = 'softmax'; outputfc = activations(net{iii},imdsValidation,layername );
myactivationsone(cv.test(iii)) =  outputfc(1,1,1,1,:);
myactivationstwo(cv.test(iii)) =  outputfc(1,1,1,2,:);

  % here gradcam should be sensivity of softmax output with respect to inputs. Looks like ADC was most influencial but very non intuitive.
  for jjj = 1:length(imdsValidation.Files)
  mylabel = classify(net{iii},readimage(imdsValidation,jjj));
  %[scoreMap,featureLayer,reductionLayer] = gradCAM(net{iii},readimage(imdsValidation,jjj),mylabel,FeatureLayer="input");
  [scoreMap,featureLayer,reductionLayer] = gradCAM(net{iii},readimage(imdsValidation,jjj),mylabel);
  niiinfo = niftiinfo(imdsValidation.Files{jjj});
  niiinfo.ImageSize = niiinfo.ImageSize(1:3);
  niiinfo.PixelDimensions = niiinfo.PixelDimensions (1:3);
  myfilepath = replace(imdsValidation.Files{jjj},'adcnetwork','newmap');
  pathsplit = split(myfilepath,'.');
  command = sprintf('mkdir -p %s',pathsplit{1});
  status = system(command);
  niftiwrite(single(scoreMap),sprintf('%s/gradcam.nii',pathsplit{1}),niiinfo );
  %sprintf('c3d -mcs %s -o label.nii -pop -o t2.nii -pop -o adc.nii', niiinfo.Filename)
  end 

end

accuracy(idweight,idepoch ) = sum(YPred == imds.Labels)/numel(imds.Labels)
C = confusionmat(YPred ,imds.Labels)
end
end

figure(2)
plot(hyperepoch,accuracy(1,:) )

figure(3)
plot(myactivationsone,imds.Labels, '+')
figure(4)
histogram(myactivationsone(imds.Labels=='0'),20)
hold on
histogram(myactivationsone(imds.Labels=='1'),20)

table(imds.Files,myactivationsone,myactivationstwo)
%% 
% _Copyright 2018 The MathWorks, Inc._
function imagedata= mycustomreader(filename)
     vectorimage = squeeze(niftiread(filename));
     %imagedata = vectorimage(:,:,:,2).* (vectorimage(:,:,:,3) >=2);
     %imagedata = vectorimage(:,:,:,2).* (vectorimage(:,:,:,4) > 0);
     %imagedata = vectorimage(:,:,:,1) + 100* vectorimage(:,:,:,3) ;
     imagedata = vectorimage;
end

function stop = stopTraining(info)
%info.ValidationAccuracy
  global foldmaxaccuracy;
  if (info.ValidationAccuracy>foldmaxaccuracy)
     foldmaxaccuracy= info.ValidationAccuracy
  end
  if (info.Epoch>40)
    stop = info.ValidationAccuracy >= foldmaxaccuracy & foldmaxaccuracy > 70 ;
  else
    stop = info.ValidationAccuracy > 90;
  end
end
