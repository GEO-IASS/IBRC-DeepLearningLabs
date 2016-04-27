function [pim,dim ] = MakeDichromatIms(im,plot_result)
%function [pim,dim ] = MakeDichromatIms(im)
% Takes an image 'im' as input and outputs it as how a protan dichromat and
% a deutan dichromat would perceive it.

if nargin < 2
    plot_result=true;
end

% Convert uint8 to double
im = double(im);
m=size(im,1);
n=size(im,2);
im = (im/255).^(2.2);

% Reduce colour domain represented
pim = 0.992052.*im + 0.003974; %Protanopes
dim = 0.957237.*im + 0.0213814; %Deuteranopes

%Transform RGB to LMS
M=[[17.8824 43.5161 4.11935];
    [3.45565 27.1554 3.86714];
    [0.0299566 0.184309 1.46709]];
%Projection matrices from normal to dichromat's vision
P=[[0 2.02344 -2.52581];
    [0 1 0];
    [0 0 1]];
D=[[1 0 0];
    [0.494207 0 1.24827];
    [0 0 1]];

%Apply transformations to each pixel for both cases
PT=M\(P*M);
DT=M\(D*M);
pim=reshape(pim,[m*n,3]);
dim=reshape(dim,[m*n,3]);
for i=1:m*n
    pim(i,:)=(PT*pim(i,:)')';
    dim(i,:)=(DT*dim(i,:)')';
end

pim=reshape(pim,[m,n,3]);
dim=reshape(dim,[m,n,3]);

im=(im.^(1/2.2))*255;
pim=(pim.^(1/2.2))*255;
dim=(dim.^(1/2.2))*255;

if plot_result
    montage({im/255, pim/255, dim/255}, 'Size', [1, 3]);
end

end

