tic
% define  folder
Folder = fullfile('C:\Users\Neel\Desktop\DLProject\data', 'caltech256'); 
disp('steps to output')

% Create ImageDatastore of the dataset for processing in Matlab.
rootFolder = fullfile(Folder, '256_ObjectCategories');
imds = imageDatastore(fullfile(rootFolder), 'LabelSource', 'foldernames','IncludeSubfolders',true);
clear Folder rootFolder;

% Split each label Using 30 images for training and rest for testing
[trainingSet, testingSet] = splitEachLabel(imds, 30);

% Load resnet
disp('1. Loading Pretrained Resnet');
 net = resnet101;
 
 %set imageSize according to DCNN first input layer
 imageSize = net.Layers(1).InputSize;
 
 disp('2.Data Augmentatiojn');
%Define Augmentation function arguments
imageAugmenter = imageDataAugmenter('RandXReflection',true,...
    'RandYReflection',false,...
    'RandRotation',[0 0],...
    'RandScale',[1 1],...
    'RandXTranslation',[0 0],...
    'RandYTranslation',[0 0]);

%Store Augmented training image into Imagedatastore
augImdstraining = augmentedImageDatastore(imageSize,trainingSet, ...
   'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);

%Store Augmented testing image into Imagedatastore
augImdstesting = augmentedImageDatastore(imageSize,testingSet, ...
   'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
%resetgpu
gpuDevice(1)
 
% Get features from resnet
 disp('3. Loading Resnet train features');
 %Extract training features from resnet101 DCNN
 resnet_features_train = activations(net,augImdstraining,'fc1000','MiniBatchSize',200);
 
 disp('4. Loading Resnet test features');
 %Extract testing features from resnet101 DCNN
 resnet_features_test = activations(net,augImdstesting,'fc1000','MiniBatchSize',200);
 
 %Reshape training and testing features from resnet101
 resnet_features_train = reshape(resnet_features_train,[1*1*1000,size(resnet_features_train,4)])' ;
 resnet_features_test = reshape(resnet_features_test,[1*1*1000,size(resnet_features_test,4)])';

 disp('5. Loading Pretrained inceptionv3');
% Load inceptionv3 
net = inceptionv3;

%set imageSize according to DCNN first input layer
imageSize = net.Layers(1).InputSize;

%Store Augmented training image into Imagedatastore
augImdstraining = augmentedImageDatastore(imageSize,trainingSet, ...
   'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);

%Store Augmented testing image into Imagedatastore
augImdstesting = augmentedImageDatastore(imageSize,testingSet, ...
   'ColorPreprocessing','gray2rgb' ,'DataAugmentation',imageAugmenter);

%resetgpu
gpuDevice(1)

disp('6. Loading inceptionv3 train features');
 %Extract training features from inceptionv3 DCNN
inceptionv3_features_train = activations(net,augImdstraining,'avg_pool','MiniBatchSize',200);

disp('7. Loading inceptionv3 test features');
 %Extract testing features from inceptionv3 DCNN
inceptionv3_features_test = activations(net,augImdstesting,'avg_pool','MiniBatchSize',200);

%Reshape training and testing features from inceptionv3
inceptionv3_features_train = reshape(inceptionv3_features_train,[1*1*2048,size(inceptionv3_features_train,4)])' ;
inceptionv3_features_test = reshape(inceptionv3_features_test,[1*1*2048,size(inceptionv3_features_test,4)])';

disp('8. Loading Pretrained googlenet');
% Load inceptionv3 
net = googlenet;

%set imageSize according to DCNN first input layer
imageSize = net.Layers(1).InputSize;

%Store Augmented training image into Imagedatastore
augImdstraining = augmentedImageDatastore(imageSize,trainingSet, ...
   'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);

%Store Augmented testing image into Imagedatastore
augImdstesting = augmentedImageDatastore(imageSize,testingSet, ...
   'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
%resetgpu
gpuDevice(1)

 disp('9. Loading googlenet train features');
 %Extract training features from googlenet DCNN
  googlenet_features_train = activations(net,augImdstraining,'loss3-classifier','MiniBatchSize',200);

  disp('10. Loading googlenet test features');
  %Extract testing features from googlenet DCNN
googlenet_features_test = activations(net,augImdstesting,'loss3-classifier','MiniBatchSize',200);

%Reshape training and testing features from googlenet
googlenet_features_train = reshape(googlenet_features_train,[1*1*1000,size(googlenet_features_train,4)])' ;
googlenet_features_test = reshape(googlenet_features_test,[1*1*1000,size(googlenet_features_test,4)])';

disp('11. Loading Pretrained densenet201');
% Load densenet201
net = densenet201;

%set imageSize according to DCNN first input layer
imageSize = net.Layers(1).InputSize;

%Store Augmented training image into Imagedatastore
augImdstraining = augmentedImageDatastore(imageSize,trainingSet, ...
   'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);

%Store Augmented testing image into Imagedatastore
augImdstesting = augmentedImageDatastore(imageSize,testingSet, ...
   'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);

%resetgpu
gpuDevice(1)

 disp('12. Loading densenet train features');
 %Extract training features from densenet DCNN
  densenet_features_train = activations(net,augImdstraining,'fc1000','MiniBatchSize',200);
 
  disp('13. Loading densenet test features');
  %Extract testing features from densenet DCNN
densenet_features_test = activations(net,augImdstesting,'fc1000','MiniBatchSize',200);

%Reshape training and testing features from densenet
densenet_features_train = reshape(densenet_features_train,[1*1*1000,size(densenet_features_train,4)])' ;
densenet_features_test = reshape(densenet_features_test,[1*1*1000,size(densenet_features_test,4)])';


disp('14. Combining the training features from All DCNN');
% Merge Resnet and googlenet deep features for training
x = horzcat(resnet_features_train,googlenet_features_train);
% Merge densenet and inceptionv3 deep features for training
w = horzcat(inceptionv3_features_train, densenet_features_train);
% Merge all deep features for training
new_F_train = horzcat(x,w);

disp('15. Combining the testing features from All DCNN');
% Merge Resnet and googlenet deep features for testing
y = horzcat(resnet_features_test,googlenet_features_test);
% Merge inceptionv3 and densenet deep features for testing
z = horzcat(inceptionv3_features_test, densenet_features_test);
% Merge all deep features for testing
new_F_test = horzcat(y,z);


%Get Train Label from training dataset
train_labels = grp2idx(trainingSet.Labels);

%Get Test Label from testing dataset
test_labels = grp2idx(testingSet.Labels);

disp('16. creating training and testing dataset for elm');
%Give labels to training and testing features
 training = horzcat(train_labels,new_F_train);
 testing = horzcat( test_labels,new_F_test);
 

C = 2^-10;
% disp('17. Classification using ELM');
% [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM(training, testing, 1, 10000, 'sig', C);

%define Number subnetworknode for MFB_ELM
number_subnetwork_node = 6;

disp('18. Classification using MFB_ELM')
[train_time,  train_accuracy,test_accuracy]=MFB_ELM(training,testing,1,1,'sig',number_subnetwork_node,C);
fprintf('Training Time = %f\n',train_time);
fprintf('Training Accuracy = %f\n',train_accuracy);
fprintf('Testing Accuracy = %f\n',test_accuracy);



timeElapsed = toc;
fprintf('Total Time Elapsed = %f\n',timeElapsed);