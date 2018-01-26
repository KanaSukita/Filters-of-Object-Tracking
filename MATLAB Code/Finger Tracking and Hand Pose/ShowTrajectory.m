function ShowTrajectory(GroundTruth, result)
figure;
plot(GroundTruth(1,:), GroundTruth(2,:), '-', 'Color', 'b', 'MarkerSize', 5);
hold on;
plot(result(1,:), result(2,:), '-.', 'Color', 'r', 'MarkerSize', 5);
hold off;
end