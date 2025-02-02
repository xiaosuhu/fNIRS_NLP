%% Create the stim base with only the current word

design = readtable('HH_time_matrix.csv');
stim_time = design.Start_time_adjusted;

t = 1/5:1/5:836;

stim_base = zeros(4180,768);
for i = 1:768
    stim_base(round(stim_time*5),i)=activations(:,i);
end

%% Create the lagged stim

% Assuming `activation` is a matrix of size Nx768 (e.g., 2095x768)
for i = 1:20
    % Create the shifted matrix and pad with zeros
    shifted_matrix = [activations(i:end, :); zeros(i-1, 768)];
    
    % Assign the result to a dynamic field in the struct
    activation_set{i} = shifted_matrix;
end


%% Create design matrix for both
for i = 1:20
design_mat{i} = convlution_canonical(stim_base, activation_set{i});
end

%% Concatenate
% d<=7
design_mat_1 = design_mat{1};

design_mat_2 = [design_mat{2}];
design_mat_3 = [design_mat{2} design_mat{3}];
design_mat_4 = [design_mat{2} design_mat{3} design_mat{4}];
design_mat_5 = [design_mat{2} design_mat{3} design_mat{4} design_mat{5}];
design_mat_6 = [design_mat{2} design_mat{3} design_mat{4} design_mat{5} design_mat{6}];
design_mat_7 = [design_mat{2} design_mat{3} design_mat{4} design_mat{5} design_mat{6} design_mat{7}];
design_mat_8 = [design_mat{2} design_mat{3} design_mat{4} design_mat{5} design_mat{6} design_mat{7} design_mat{8}];


% d>7
inc = 2;
design_mat_8 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 3;
design_mat_9 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 4;
design_mat_10 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 5;
design_mat_11 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 6;
design_mat_12 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 7;
design_mat_13 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 8;
design_mat_14 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 9;
design_mat_15 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];
inc = 10;
design_mat_16 = [design_mat{1+inc} design_mat{2+inc} design_mat{3+inc} design_mat{4+inc} design_mat{5+inc} design_mat{6+inc} design_mat{7+inc}];


%% Run PCA to get the most important stims 
for i = 1:16 % Assume you have design_mat_1 to design_mat_10
    eval(strcat('pc_mat_',num2str(i) ,'= pca_20(design_mat_',num2str(i),');'));
end