function [A_b] = sysBuildPAT_mod_Band(object,indxi,indyi,time,medium,sensor,sensor_radius)
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generate system matrix%%%%%%%%%%%%%%%%%%%%%
Nxi = length(indxi);
Nyi = length(indyi);
tl = time.length;
sml = length(sensor.mask);
Nx = tl*sml;

xx =-100;
yy =-100;
% xx = 0;
% % yy = 0;
% 
c_x = ceil(object.Nx/2)+1; c_y = ceil(object.Ny/2)+1;
object.p0 = zeros(object.Nx, object.Ny);
object.p0(c_x+xx,c_y+yy) = 1;
% 
% sd = forward(object, time, medium, sensor);

sensor.frequency_response = [2.25e6 70];
sd_b = forward(object, time, medium, sensor);
% sensor/receiver position
x_receive = sensor.mask(1,:);
y_receive = sensor.mask(2,:);

x_img = ((1:object.Nx)-((object.Nx+1)/2))*object.x/(object.Nx-1); x_img = repmat(x_img',1,object.Ny);
y_img = ((1:object.Ny)-((object.Ny+1)/2))*object.y/(object.Ny-1); y_img = repmat(y_img,object.Nx,1);

% sd2 = zeros(size(sd));
sd2_b = zeros(size(sd_b));
% A = zeros(Nx,Nxi*Nyi);
A_b = zeros(Nx,Nxi*Nyi);

sensor_distance = sqrt((x_img(c_x+xx,c_y+yy)-x_receive).^2 +(y_img(c_x+xx,c_y+yy)-y_receive).^2);

for i = 1:Nxi
    %i
    for j = 1:Nyi
        x = x_img(indyi(j),indxi(i)); y = y_img(indyi(j),indxi(i));
        r = sqrt((x_receive-x).^2+(y_receive-y).^2);
        r1 = abs(r-sensor_distance);
        ind1 = find(r1<1e-6);
        r1(ind1) = 0;
        ind = ceil(r1/medium.sound_speed/time.dt);
        for k = 1:sml
%             sd1 = sd(k,:)*sqrt(sensor_distance(k)/r(k));
            sd1_b = sd_b(k,:)*sqrt(sensor_distance(k)/r(k));
            if(r(k)>= sensor_distance(k))
%                 sd2(k,:) = circshift(sd1,[0 ind(k)]);
                sd2_b(k,:) = circshift(sd1_b,[0 ind(k)]);
            else
%                 sd2(k,:) = circshift(sd1,[0 -ind(k)]);  
                sd2_b(k,:) = circshift(sd1_b,[0 -ind(k)]);
            end
        end
%         A(:,(i-1)*Nyi+j) = reshape(sd2,Nx,1); 
        A_b(:,(i-1)*Nyi+j) = reshape(sd2_b,Nx,1); 
    end
end
