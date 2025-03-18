function HZ = Get_HZ(Z, Ac, Ar, picsize)
% Compute HZ in blurring case

n = size(Z, 2);
HZ = [];
for k = 1 : n
    hz = Ac*reshape(Z(:,k), picsize)*Ar';
    HZ = [HZ hz(:)];
end