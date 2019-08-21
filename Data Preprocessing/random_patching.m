clc; clear;
addpath('./Limited_Bandwidth_Neg_01_Left/');
addpath('./Limited_Bandwidth_Neg_01_Left_noise_interpolated_mix_data/');

% patch parameters
patch_size = 64;
stride_detectors = 20;
stride_sample_time = 32;

% detector parameterss
n_samples = 2851;
detectors = 200;
sampling_time = 512;
start = 1;
% snr = [10, 20, 30, 40, 60];
save_path = '/run/user/1000/gvfs/sftp:host=nvidia-dgx.serc.iisc.ac.in,user=cdsvenk/localscratch/cdsvenk/Gaurav/Train_w_net/';
for i=start:n_samples
    
    limited = strcat('Limited_noise_interpolated', num2str(i), '.mat');
%     disp(limited);
    load(limited);
    
    limited_truth = strcat('Limited', num2str(i), '.mat');
    disp(limited_truth);
    load(limited_truth);
    
    full = strcat('Full', num2str(i), '.mat');
%     disp(full);
    load(full);
    
%     figure;imshow(sdn2_v_left_full, []);
%     figure;imshow(limited_noise_interpolated, []);
    
    m = 1;
    n=1;
    k=1;
    count =0;
    while (m <= detectors - patch_size +1)
        n=1;
        while(n<=sampling_time - patch_size +1)
            lim_patch = limited_noise_interpolated(m:m+patch_size-1, n:n+patch_size-1);
            full_patch = sdn2_v_left_full(m:m+patch_size-1, n:n+patch_size-1);
            lim_truth_patch = sdn2_v_left_limited(m:m+patch_size-1, n:n+patch_size-1);
            count = count +1;
            save_name = strcat(save_path, 'Limited_noise_interpolated', num2str(i), '_', num2str(count));
            save (save_name, 'lim_patch');
            save_name = strcat(save_path, 'Limited', num2str(i), '_', num2str(count));
            save (save_name, 'lim_truth_patch');
            save_name = strcat(save_path, 'Full', num2str(i), '_', num2str(count));
            save (save_name, 'full_patch');
%             disp('saved');
            n = n + stride_sample_time;
%             figure;imshow(lim_patch, []);
%             figure;imshow(full_patch, []);
            
        end
        m  = m + stride_detectors;
    end
end