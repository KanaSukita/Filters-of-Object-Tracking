function f = GetFunc(net)
inxoffset = net.inputs{1,1}.processSettings{1,1}.xoffset;
ingain = net.inputs{1,1}.processSettings{1,1}.gain;
inymin = net.inputs{1,1}.processSettings{1,1}.ymin;
outymin =  net.outputs{2}.processSettings{1,1}.ymin;
outgain =  net.outputs{2}.processSettings{1,1}.gain;
outxoffset =  net.outputs{2}.processSettings{1,1}.xoffset;
w1 = net.iw{1,1};  
b1 = net.b{1,1};        
w2 = net.lw{2,1};      
b2 = net.b{2,1};        
f=@(x)((w2*(2./(1+exp(-2*(w1*((x-inxoffset).*ingain+inymin) + b1)))-1) + b2)-outymin)./outgain+outxoffset;
end