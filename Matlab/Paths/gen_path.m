function route = gen_path(shape, options)
arguments
    shape (1, 1) string
    options.f (1, 1) {mustBePositive} = 1; % frequency for sin function
end

switch shape
    case "Sinus"
        x = linspace(0,1,200);
        y = sin(2*pi*options.f*x);
    case "Straight"
        x = linspace(0,10,1000);
        y = 0*x;
    case "Sample"
        route = load('route.mat').route;
        return
    case "8"        
        route = load('eight.mat').eight;        
        return
end

route = [x' y'];
