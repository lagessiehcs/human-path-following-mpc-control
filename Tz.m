function T = Tz(rad, t)

    tol = eps;
    
    ct = cos(rad);
    st = sin(rad);
    
    % Make almost zero elements exactly zero
    if abs(st) < tol
        st = 0;
    end
    if abs(ct) < tol
        ct = 0;
    end

    
    % Create the homogenous transfomation matrix
    T = [
        ct  -st  0  t(1)
        st   ct  0  t(2)
        0    0   1   0
        0    0   0   1
        ];
end