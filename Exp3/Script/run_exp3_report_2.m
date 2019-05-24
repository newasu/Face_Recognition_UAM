
exp3_A_pelm2_distance(1) = Exp3_A_pelm2_0001_distance_random_select.testResult.score{1};
exp3_A_pelm2_distance(2) = Exp3_A_pelm2_0005_distance_random_select.testResult.score{1};
exp3_A_pelm2_distance(3) = Exp3_A_pelm2_001_distance_random_select.testResult.score{1};
exp3_A_pelm2_distance(4) = Exp3_A_pelm2_005_distance_random_select.testResult.score{1};
exp3_A_pelm2_distance(5) = Exp3_A_pelm2_01_distance_random_select.testResult.score{1};
exp3_A_pelm2_distance(6) = Exp3_A_pelm2_02_distance_random_select.testResult.score{1};

exp3_A_pelm2_mean(1) = Exp3_A_pelm2_0001_mean_random_select.testResult.score{1};
exp3_A_pelm2_mean(2) = Exp3_A_pelm2_0005_mean_random_select.testResult.score{1};
exp3_A_pelm2_mean(3) = Exp3_A_pelm2_001_mean_random_select.testResult.score{1};
exp3_A_pelm2_mean(4) = Exp3_A_pelm2_005_mean_random_select.testResult.score{1};
exp3_A_pelm2_mean(5) = Exp3_A_pelm2_01_mean_random_select.testResult.score{1};
exp3_A_pelm2_mean(6) = Exp3_A_pelm2_02_mean_random_select.testResult.score{1};

exp3_A_pelm2_multiply(1) = Exp3_A_pelm2_0001_multiply_random_select.testResult.score{1};
exp3_A_pelm2_multiply(2) = Exp3_A_pelm2_0005_multiply_random_select.testResult.score{1};
exp3_A_pelm2_multiply(3) = Exp3_A_pelm2_001_multiply_random_select.testResult.score{1};
exp3_A_pelm2_multiply(4) = Exp3_A_pelm2_005_multiply_random_select.testResult.score{1};
exp3_A_pelm2_multiply(5) = Exp3_A_pelm2_01_multiply_random_select.testResult.score{1};

exp3_A_pelm2_sum(1) = Exp3_A_pelm2_0001_sum_random_select.testResult.score{1};
exp3_A_pelm2_sum(2) = Exp3_A_pelm2_0005_sum_random_select.testResult.score{1};
exp3_A_pelm2_sum(3) = Exp3_A_pelm2_001_sum_random_select.testResult.score{1};
exp3_A_pelm2_sum(4) = Exp3_A_pelm2_005_sum_random_select.testResult.score{1};
exp3_A_pelm2_sum(5) = Exp3_A_pelm2_01_sum_random_select.testResult.score{1};

exp3_A_pelm2 = [exp3_A_pelm2_distance; exp3_A_pelm2_mean; ...
    exp3_A_pelm2_multiply; exp3_A_pelm2_sum];

x_axis = [0.1 0.5 1 5 10];

markerSize = 3;

newFigure;

plot(x_axis, exp3_A_pelm2_distance, '-o', 'MarkerSize', markerSize);
hold on;

plot(x_axis, exp3_A_pelm2_mean, '-o', 'MarkerSize', markerSize);
hold on;

plot(x_axis, exp3_A_pelm2_multiply, '-o', 'MarkerSize', markerSize);
hold on;

plot(x_axis, exp3_A_pelm2_sum, '-o', 'MarkerSize', markerSize);
hold on;

legend({'PELM\_distance', 'PELM\_mean', 'PELM\_multiply', 'PELM\_sum'}, ...
    'Location', 'east');

