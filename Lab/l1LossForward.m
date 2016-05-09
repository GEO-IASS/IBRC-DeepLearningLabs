function y = l1LossForward(x,r)
% TODO: Replace the following line with your implementation
y=sum(abs(x(:)-r(:)));

y = y / (size(x,1) * size(x,2)) ;  % normalize by image size
