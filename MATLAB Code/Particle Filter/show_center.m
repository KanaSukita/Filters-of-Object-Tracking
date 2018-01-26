function show_particles(X, Y_k)

figure(1)
imshow(Y_k)
title('+++ Showing Particles +++')

hold on
x_center = mean(X(2,:));
y_center = mean(X(1,:));
plot(x_center, y_center, '+', 'MarkerEdgeColor', 'y', 'MarkerSize', 50)
hold off

drawnow
end