clc; clear;
addpath('./Limited_Bandwidth_Neg_01_Left');
n_samples = 2851;
detectors = 200;
sampling_time = 512;
snr = [10, 20, 30, 40, 50, 60, 70];
save_path = 'Limited_Bandwidth_Neg_01_Left_noise_interpolated_mix_data/';
start = 1;
for i=start:n_samples
    
    limited_sino = strcat('Limited', num2str(i), '.mat');
    disp(limited_sino);
    load(limited_sino);
    %figure;imshow(sdn2_v_left_limited,[]);
    
    
    %%% add noise
    snr_index = randi(7);
    sdn_without_noise = sdn2_v_left_limited;
    sdn_noise = addNoise(sdn_without_noise, snr(snr_index), 'peak');
    %figure;imshow(sdn_noise,[]);
    
    %%% Downsample
    down_sinogram = zeros(detectors/2, sampling_time);
    k=1;
    for j=1:detectors
        if (mod(j,2)==0)
            down_sinogram(k,:) = sdn_noise(j,:);
            k=k+1;
        end
    end
    %figure;imshow(down_sinogram,[]);
     
    %%% Upsample (Interpolate using nearest neighbour)
    limited_noise_interpolated = imresize(down_sinogram, [detectors, sampling_time], 'nearest');
    %figure;imshow(Limited_noise_interpolated,[]);
    
    %%% Save 
    save_name = strcat(save_path, 'Limited_noise_interpolated', num2str(i), '.mat');
    save (save_name, 'limited_noise_interpolated');
end



%% 
% pred_patch69 = pred_pad(200-64: 200, 230:230+64);
% limited_noise69_patch = limited_noise_interpolated(200-64: 200, 230:230+64);

r=1;
eps = 1e-3;
alpha = 1.05;
betaa = 1.05;
gap_w = 0.1;
gap_h = 0.1;
img_rows = 1;
img_cols = 1;
figure_size = 40; 
aspect_x = 1;
aspect_y = 1;
save1 = figure;
min1 = 0;
max1 = 1;




ha = tightPlots(img_rows, img_cols, figure_size, [aspect_x aspect_y], [gap_w gap_h], [0 0], [0 0], 'centimeters');

axes(ha(1)); imshow(pred_patch69,[]); 
% c = gray;c = flipud(c);colormap(c);
% axis image; axis off; tx=text(170,190, '(a)');tx.FontWeight = 'bold';tx.FontSize = 20;
% tx=text(-12,160, 'SNR = 20 dB');tx.FontWeight = 'bold';tx.Rotation = 90;tx.FontSize = 20;
% line(15:65,195*ones(51),'Color','k','LineStyle','-','Linewidth',3);
% line(15*ones(51),145:195,'Color','k','LineStyle','-','Linewidth',3);
% tx=text(25,185, '5 mm','color','k');tx.FontWeight = 'bold';
% tx.FontSize = 16;
saveas(save1,'pred_patch69.png');


    