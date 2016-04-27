function [ w,b ] = TrainSVM( Xtrain, y, lambda, nepochs,verbose)
%TRAINSVM function [ w,b ] = TrainSVM( Xtrain, y, lambda, nepochs )
%   Takes train data Xtrain and labels ys and trains a SVM

if nargin<5
    verbose=0;
end
if nargin<4
    nepochs=40;
end
if nargin<3
    lambda=0.0001;
end
[D,N]=size(Xtrain);

if verbose>0
disp(['Training SVM with ',int2str(nepochs),' epochs']);
end

b=0;
w=zeros(D,1);

t=1;

for n=1:nepochs
    
    class=zeros(N,1);
    for i=randperm(N)
        nt=1/lambda/t;
        class(i)=y(i)*(w'*Xtrain(:,i)+b)<1;
        if class(i)
            w=(1-nt*lambda)*w+nt*y(i)*Xtrain(:,i);
            b=b+nt*y(i);
        else
            w=(1-nt*lambda)*w;
        end
        t=t+1;
    end
    a=min([1,1/norm(w)/sqrt(lambda)]);
    w=a*w;
    b=a*b;
    if verbose>0
        disp(['Epoch #',int2str(n)])
    end
    if verbose>1
        acc=1-sum(class)/N;
        disp(['Acc: ',num2str(acc)])
    end
end
if verbose>0
    disp('Training done')
end

end

