%% Load data
addpath(genpath('D:\test_image'))
addpath(genpath('libsvm-3.22'))
addpath('functions')
addpath('splitBregmanROF_mex')
load('Indian_pines.mat')
load('Indian_pines_gt.mat');
% load('paviaU.mat')
% load('PaviaU_gt.mat');
% load('Salinas.mat')
% load('Salinas_gt.mat');
% load('indian_pines2010.mat')
% % load('indian_pines2010_gt.mat')
% load('DFC2018_Houston.mat')
% load('DFC2018_Houston_gt.mat')
img = double(indian_pines);
labels = double(indian_pines_gt) ;
% img = double(salinas);
% labels = double(salinas_gt);
%% size of image  
[no_lines, no_rows, no_bands] = size(img);  
img2=average_fusion(img,15);
 %% normalization
no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);

%% ATV Structural feature extraction
tic;
 fimg1 = ATV_test(fimg,0.004,2,0.02);
 t_img1 = toc;
 tic;
 fimg2 = ATV_test(fimg,0.02,2,0.02);        
 t_img2 = toc;
 tic;
 fimg3 = ATV_test(fimg,0.01,2,0.02);
 t_img3 = toc;
 f_fimg=cat(3,fimg1,fimg2,fimg3);
 
%% ITV Globe Smoothing 
tic;
 fimg_v = ITV_test(f_fimg,20,100);
 t_itv = toc;

%% SVM classification
tic;
fimg = ToVector(fimg_v);
fimg = fimg';
fimg=double(fimg);
OA1=[];AA1=[];Kappa1=[];CA1=[];
indexes = [];
% train_number =[6,7,6,6,6,6,6,7,6,7,8,6,6,6,6,7];
no_classes = 16;
train_number = ones(1,no_classes)*10;

for flag = 1:10
    
        [train_SL,test_SL,index]= GenerateSample(labels,train_number,no_classes);
        shuffle_train = randperm(size(train_SL,2));
        shuffle_test = randperm(size(test_SL,2));
        train_SL = train_SL(:,shuffle_train);
        test_SL = test_SL(:,shuffle_test);
        train_samples = fimg(:,train_SL(1,:))';
        train_labels= train_SL(2,:)';
        test_samples = fimg(:,test_SL(1,:))';
        test_labels = test_SL(2,:)';
        test_id = test_SL(1,:);
        train_id = train_SL(1,:);
        indexes = [indexes index];
        [train_samples,M,m] = scale_func(train_samples);
        [fimg11 ] = scale_func(fimg',M,m);
        % Selecting the paramter for SVM
        [Ccv, Gcv, cv, cv_t]=cross_validation_svm(train_labels,train_samples);
        % Training using a Gaussian RBF kernel
        parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
        model=svmtrain(train_labels,train_samples,parameter);
        % Evaluation
        %1
        [predict_label, accuracy, P1] = svmpredict(ones(no_lines*no_rows,1),fimg11,model); 
        [OA1(flag),Kappa1(flag),AA1(flag),CA1(:,flag)] = calcError(test_SL(2,:)'-1,predict_label(test_id)-1,[1:no_classes]);
       
end
t_svm = toc;

 %% Evaluation
oa_mean = mean(OA1);
aa_mean = mean(AA1);
kappa_mean = mean(Kappa1);

