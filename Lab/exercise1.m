setup() ;

%% Part 1.1: convolution

%% Part 1.1.1: convolution by a single filter
%%
% Load an image and convert it to gray scale and single precision
x = im2single(rgb2gray(imread('data/ray.jpg'))) ;

% Define a filter
w = single([
   0 -1  0
  -1  4 -1
   0 -1  0]) ;

% Apply the filter to the image
y = vl_nnconv(x, w, [],'Pad',1) ;

%%
% Visualize the results
figure(11) ; clf ; colormap gray ;
set(gcf, 'name', 'Part 1.1: convolution') ;

subplot(2,2,1) ;
imagesc(x) ;
axis off image ;
title('Input image x') ;

subplot(2,2,2) ;
imagesc(w) ;
axis off image ;
title('Filter w') ;

subplot(2,2,3) ;
imagesc(y) ;
axis off image ;
title('Output image y') ;

%% Part 1.1.2: convolution by a bank of filters

% Concatenate three filters in a bank
w1 = single([
   0 -1  0
  -1  4 -1
   0 -1  0]) ;

w2 = single([
  -1 0 +1
  -1 0 +1
  -1 0 +1]) ;

w3 = single([
  -1 -1 -1
   0  0  0
  +1 +1 +1]) ;

w4 = single([
  -1 0 +1
  -2 0 +2
  -1 0 +1]) ;

w5 = single([
  -1 -2 -1
   0  0  0
  +1 +2 +1]) ;

wbank = cat(4, w1, w2, w3, w4, w5) ;

% Apply convolution
y = vl_nnconv(x, wbank, []) ;

%% Show feature channels
figure(12) ; clf('reset') ;
set(gcf, 'name', 'Part 1.1.2: channels') ;
colormap gray ;
showFeatureChannels(y) ;

%% Part 1.1.3: convolving a batch of images
%%
x1 = im2single(rgb2gray(imread('data/ray.jpg'))) ;
x2 = im2single(rgb2gray(imread('data/crab.jpg'))) ;
x = cat(4, x1, x2) ;

y = vl_nnconv(x, wbank, []) ;
%% visualize the results
figure(13) ; clf('reset') ;
set(gcf, 'name', 'Part 1.1.3: channels, ray image') ;
colormap gray ;
showFeatureChannels(y(:,:,:,1));

figure(14) ; clf('reset') ;
set(gcf, 'name', 'Part 1.1.3: channels, crab image') ;
colormap gray ;
showFeatureChannels(y(:,:,:,2));
%% Part 1.2: non-linear activation functions (ReLU)

%% Part 1.2.1: Laplacian and ReLU
%%
x = im2single(rgb2gray(imread('data/ray.jpg'))) ;

% Convolve with the negated Laplacian
y = vl_nnconv(x, - w, []) ;

t=0.1;
% Apply the ReLU operator
z = vl_nnrelu(y) ;
z2 = 1./(1+exp(-y));
z3=tanh(y);
%% visualize the results
figure(15) ; clf('reset') ;
set(gcf, 'name', 'Part 1.2.1: Original, Laplacian and ReLU') ;
colormap gray ;
subplot(131);imagesc(x);axis image;
subplot(132);imagesc(y);axis image;
subplot(133);imagesc(z);axis image;
%%
figure(16) ; clf('reset') ;
set(gcf, 'name', 'Part 1.2.1: ReLU, Sigmoid and Tanh') ;
colormap gray ;
subplot(131);imagesc(z);axis image;
subplot(132);imagesc(z2);axis image;
subplot(133);imagesc(z3);axis image;

%% Part 1.2.2: effect of adding a bias
%%
bias = single(- 0.2) ;
y = vl_nnconv(x, - w, bias) ;
z = vl_nnrelu(y) ;

%% visualize the results
figure(17) ; clf('reset') ;
set(gcf, 'name', 'Part 1.2.2: ReLU with bias') ;
colormap gray ;
subplot(131);imagesc(x);axis image;
subplot(132);imagesc(y);axis image;
subplot(133);imagesc(z);axis image;

%% Max avg pooling
mp1=vl_nnpool(x,4,'Method','max');
mp2=vl_nnpool(x,8,'Method','max');
mp3=vl_nnpool(y,4,'Method','max');
mp4=vl_nnpool(y,8,'Method','max');

ap1=vl_nnpool(x,4,'Method','avg');
ap2=vl_nnpool(x,8,'Method','avg');
ap3=vl_nnpool(y,4,'Method','avg');
ap4=vl_nnpool(y,8,'Method','avg');

%% %% visualize the results
figure(17) ; clf('reset') ;
set(gcf, 'name', 'Part 1.2.2: ReLU with bias') ;
colormap gray ;
subplot(241);imagesc(mp1);axis image;
subplot(242);imagesc(mp2);axis image;
subplot(243);imagesc(mp3);axis image;
subplot(244);imagesc(mp4);axis image;

subplot(245);imagesc(ap1);axis image;
subplot(246);imagesc(ap2);axis image;
subplot(247);imagesc(ap3);axis image;
subplot(248);imagesc(ap4);axis image;
