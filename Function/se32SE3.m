function [ SE3 ] = se32SE3( se3 )
    %se3_SE3 Exponential Mapping from Lie Algebra to Lie Group
    %   se3 is a 1x6 Column Vector of the form=[v1 v2 v3 w1 w2 w3] which is
    %   defined using 6 Generator Matrices(4x4)
    %   each of the six elements on multiplication with the generator matrices
    %   as follows give the complete matrix:
    %   se3 = v1*G1 + v2*G2 + v3*G3 + w1*G4 + w2*G5 + w3*G6
    %   To map se3 to SE3 we need to perform e^(se3)
    %   This can be done by following the algorithm:
    %   Algorithm
    %   
    %
    %
    %
        ro=se3(1:3)';
        phi=se3(4:6)';
        phi_hat=[0 -phi(3) phi(2);phi(3) 0 -phi(1);-phi(2) phi(1) 0];
        theta=sqrt(phi'*phi);
        if(theta~=0)
            A=sin(theta)/theta;
            B=(1-cos(theta))/(theta^2);
            C=(1-A)/(theta^2);
        else
            A=0;
            B=0;
            C=0;  
        end
        R=eye(3)+(A*phi_hat)+(B*(phi_hat*phi_hat));
        V=eye(3)+B*phi_hat+C*(phi_hat*phi_hat);
        Vp=V*ro;
        SE3=zeros(4);
        SE3(1:3,1:3)=R;
        SE3(1:3,4)=Vp;
        SE3(4,4)=1;
    end