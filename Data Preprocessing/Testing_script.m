%%
% load('limited_noise_pa.mat')
% figure;imshow(pred_pad, [])
clc;clear;close all;

weights = 'v5';
pred_path = strcat('./Results/' , weights, '/*.mat');
files = dir(pred_path);

for i=1:size(files)
    load(files(i).name);
    pred = pred_pad(29:28+200,:);
    pred = double(pred);
    noisy = strsplit(files(i).name,'_');
    noisy_file = strcat('./Test',noisy{1},'_',noisy{2},'_',noisy{3},'_',noisy{4},'_',noisy{5}, '.mat' );
    load(noisy_file);
    break;
    
%     file.name(i)
end

% load('Limited_noise_interpolated_bv_20_pred.mat')
% load('/home/safran/Desktop/Gaurav Jain/Test_samples/bv_sinogram1.mat')
% load('/home/safran/Desktop/Gaurav Jain/Test_data/Limited_noise_interpolated_bv_20.mat')
% pred = pred_pad(29:28+200,:);
% pred = double(pred);
% disp('original vs noisy');
% immse(limited_noise_interpolated,sdn2_v_right_1)
% disp('original vs pred');
% immse(pred,sdn2_v_right_1)
% 
% %%
% figure;
% imshow(pred,[]);
% figure;
% imshow(limited_noise_interpolated,[]);

%%
% close all

% figure()
% imshow(lim_patch,[])
% figure()
% imshow(full_patch,[])
