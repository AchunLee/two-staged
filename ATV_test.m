function S = ATV_test(I,lambda,sigma,sharpness)

% Derive from tsmooth  tsmooth(I, lambda, sigma, maxIter). FOR DETAILS AND ORIGINAL CODE PLEASE REFER PAPER :
%"Structure Extraction from Texture via Relative Total Variation", Li Xu, Qiong Yan, Yang Xia, Jiaya Jia, ACM Transactions on Graphics, 
% (SIGGRAPH Asia 2012), 2012.

%. WE HAVE IMPROVED IT.

    I = im2double(I);
    x = I;
    sigma_iter = sigma;
    lambda = lambda/2.0;
    dec=2.0;
     while sigma_iter >= 0.5
        [wx, wy] = computeTextureWeights(x, sigma_iter, sharpness);
         x = solveLinearEquation(I, wx, wy, lambda);
         sigma_iter = sigma_iter/dec;
    end
    S = x;      
end

function [retx, rety] = computeTextureWeights(fin, sigma,sharpness)

   fx = diff(fin,1,2);
   fx = padarray(fx, [0 1 0], 'post');
   fy = diff(fin,1,1);
   fy = padarray(fy, [1 0 0], 'post');
      
   vareps_s = sharpness;
   vareps = 0.001;

   wto = max(sum(sqrt(fx.^2+fy.^2),3)/size(fin,3),vareps_s).^(-1); 
   fbin = lpfilter(fin, sigma);
   gfx = diff(fbin,1,2);
   gfx = padarray(gfx, [0 1], 'post');
   gfy = diff(fbin,1,1);
   gfy = padarray(gfy, [1 0], 'post');     
   wtbx = max(sum(abs(gfx),3)/size(fin,3),vareps).^(-1); 
   wtby = max(sum(abs(gfy),3)/size(fin,3),vareps).^(-1);   
   retx = wtbx.*wto;
   rety = wtby.*wto;

   retx(:,end) = 0;
   rety(end,:) = 0;
   
end

function ret = conv2_sep(im, sigma)
  ksize = bitor(round(5*sigma),1);
  g = fspecial('gaussian', [1,ksize], sigma); 
  ret = conv2(im,g,'same');
  ret = conv2(ret,g','same');  
end

function FBImg = lpfilter(FImg, sigma)     
    FBImg = FImg;
    for ic = 1:size(FBImg,3)
        FBImg(:,:,ic) = conv2_sep(FImg(:,:,ic), sigma);
    end   
end

function OUT = solveLinearEquation(IN, wx, wy, lambda)
% 
% The code for constructing inhomogenious Laplacian is adapted from 
% the implementaion of the wlsFilter. 
% 
% For color images, we enforce wx and wy be same for three channels
% and thus the pre-conditionar only need to be computed once. 
% 
    [r,c,ch] = size(IN);
    k = r*c;
    dx = -lambda*wx(:);
    dy = -lambda*wy(:);
    B(:,1) = dx;
    B(:,2) = dy;
    d = [-r,-1];
    A = spdiags(B,d,k,k);
    e = dx;
    w = padarray(dx, r, 'pre'); w = w(1:end-r);
    s = dy;
    n = padarray(dy, 1, 'pre'); n = n(1:end-1);
    D = 1-(e+w+s+n);
    A = A + A' + spdiags(D, 0, k, k); 
    if exist('ichol','builtin')
        L = ichol(A,struct('michol','on'));    
        OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            [tout, flag] = pcg(A, tin(:),0.1,100, L, L'); 
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
    else
        OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            tout = A\tin(:);
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
    end
        
end