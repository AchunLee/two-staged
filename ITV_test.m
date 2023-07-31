function [FE]=ITV_test(Y,r_max,u)

%Derive from OTVCA. PLEASE REFER PAPER: B. Rasti, M. O. Ulfarsson, and J. R. Sveinsson, Hyperspectral Feature Extraction Using Total Variation Component Analysis?, 
% IEEE Trans. Geoscience and Remote Sensing, 54 (12), 6976-6985.


[nr1,nc1,p1]=size(Y);
RY=reshape(Y,nr1*nc1,p1);
%Y=reshape(RY,nr1,nc1,p1); % mean value recustion
%RY=RY-mean(RY);
m = min(Y(:));
M = max(Y(:));
NRY=(RY-m)/(M-m);
[~,~,V1] = svd(NRY,'econ');
V=V1(:,1:r_max);
FE=zeros(nr1,nc1,r_max);
    C1=NRY*V(:,1:r_max);
    PC=reshape(C1,nr1,nc1,r_max);
    for j = 1:r_max
        FE(:,:,j)=splitBregmanROF(PC(1:nr1,1:nc1,j),u,.1);
    end 
end