function se3 = SE32se3_back( SE3 )
    lie_hat = logm(SE3);
    rho = lie_hat(1:3,4);
    se3 = [rho;[lie_hat(3,2);lie_hat(1,3);lie_hat(2,1)]]; %6-by-1
end
