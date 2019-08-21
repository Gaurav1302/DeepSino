clc; clear;
addpath('./Test_samples/');
detectors = 200;
sampling_time = 512;

save_path = './Test_data/limited_bw/';


bv = 'bv_sinogram1.mat';
der = 'der_sinogram1.mat';
in_vivo = 'in_vivo_sinogram1.mat';
pat = 'pat_sinogram1.mat';

load(bv);
bv_sino = sdn2_v_right_1;
load(der);
der_sino = sdn2_v_right_1;

load(pat);
pat_sino = sdn2_v_right_1;

noise = 20;
for i=1:3
    load(bv);
    original = sdn2_v_right_1;
    save_name = strcat(save_path, 'original_bv.mat');
%     save (save_name, 'original');
    sdn_without_noise = sdn2_v_right_1;
    sdn_noise = addNoise(sdn_without_noise, noise, 'peak');
    down_sinogram = zeros(detectors/2, sampling_time);
    k=1;
    for j=1:detectors
        if (mod(j,2)==0)
            down_sinogram(k,:) = sdn_noise(j,:);
            k=k+1;
        end
    end
    limited_bw = down_sinogram;
    limited_noise_interpolated = imresize(down_sinogram, [detectors, sampling_time], 'nearest');
%     save_name = strcat(save_path, 'Limited_noise_interpolated_bv_', num2str(noise), '.mat');
%     save (save_name, 'limited_noise_interpolated');
    save_name = strcat(save_path, 'Limited_bw_bv_', num2str(noise), '.mat');
    save (save_name, 'limited_bw');
    
    
    
    load(der);
    original = sdn2_v_right_1;
    save_name = strcat(save_path, 'original_der.mat');
%     save (save_name, 'original');
    sdn_without_noise = sdn2_v_right_1;
    sdn_noise = addNoise(sdn_without_noise, noise, 'peak');
    down_sinogram = zeros(detectors/2, sampling_time);
    k=1;
    for j=1:detectors
        if (mod(j,2)==0)
            down_sinogram(k,:) = sdn_noise(j,:);
            k=k+1;
        end
    end
    limited_bw = down_sinogram;
    limited_noise_interpolated = imresize(down_sinogram, [detectors, sampling_time], 'nearest');
%     save_name = strcat(save_path, 'Limited_noise_interpolated_der_', num2str(noise), '.mat');
%     save (save_name, 'limited_noise_interpolated');
    save_name = strcat(save_path, 'Limited_bw_der_', num2str(noise), '.mat');
    save (save_name, 'limited_bw');
    
    load(pat);
    original = sdn2_v_right_1;
    save_name = strcat(save_path, 'original_pat.mat');
%     save (save_name, 'original');
    sdn_without_noise = sdn2_v_right_1;
    sdn_noise = addNoise(sdn_without_noise, noise, 'peak');
    down_sinogram = zeros(detectors/2, sampling_time);
    k=1;
    for j=1:detectors
        if (mod(j,2)==0)
            down_sinogram(k,:) = sdn_noise(j,:);
            k=k+1;
        end
    end
    limited_bw = down_sinogram;
    limited_noise_interpolated = imresize(down_sinogram, [detectors, sampling_time], 'nearest');
    save_name = strcat(save_path, 'Limited_bw_pat_', num2str(noise), '.mat');
    save (save_name, 'limited_bw');
    
    noise = noise + 20;
end 


load(in_vivo);
in_vivo_sino = pt_data;
down_sinogram = zeros(detectors/2, sampling_time);
k=1;
for j=1:detectors
    if (mod(j,2)==0)
        down_sinogram(k,:) = in_vivo_sino(j,:);
        k=k+1;
    end
end
limited_bw = down_sinogram;
limited_noise_interpolated = imresize(down_sinogram, [detectors, sampling_time], 'nearest');
save_name = strcat(save_path, 'Limited_bw_in-vivo', '.mat');
% save (save_name, 'limited_noise_interpolated');
save (save_name, 'limited_bw');


