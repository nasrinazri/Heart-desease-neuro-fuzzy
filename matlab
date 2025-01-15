%% Clear Workspace
close all;
clc;

%% Load Data
filename = 'C:\Program Files\MATLAB\heart_attack_prediction_dataset_Modified.xlsx'; % specify your file name
opts = detectImportOptions(filename);
opts.VariableNamingRule = 'preserve'; % Preserve original column headers
data = readtable(filename, opts);

%% Extract Features and Target
Age = data{:, 1};
Gender = categorical(data{:, 2});
Cholesterol = data{:, 3};
BloodPressure = data{:, 4};
HeartRate = data{:, 5};
Diabetes = data{:, 6};
FamilyHistory = data{:, 7};
Smoking = data{:, 8};
Obesity = data{:, 9};
Alcohol = data{:, 10};
Exercise = data{:, 11};
Diet = categorical(data{:, 12});

output = data{:, 25}; % Target variable

%% Fuzzy Logic System - Initialization
% Define input variables for fuzzy inference system
fis = mamfis('Name', 'HeartAttackRiskFIS');

% Age Fuzzy Membership
fis = addInput(fis, [20 80], 'Name', 'Age');
fis = addMF(fis, 'Age', 'trapmf', [20 20 30 40], 'Name', 'Young');
fis = addMF(fis, 'Age', 'trapmf', [30 40 50 60], 'Name', 'MiddleAged');
fis = addMF(fis, 'Age', 'trapmf', [50 60 80 80], 'Name', 'Old');

% Cholesterol Fuzzy Membership
fis = addInput(fis, [100 400], 'Name', 'Cholesterol');
fis = addMF(fis, 'Cholesterol', 'trapmf', [100 100 150 200], 'Name', 'Low');
fis = addMF(fis, 'Cholesterol', 'trapmf', [150 200 250 300], 'Name', 'Moderate');
fis = addMF(fis, 'Cholesterol', 'trapmf', [250 300 400 400], 'Name', 'High');

% Blood Pressure Fuzzy Membership
fis = addInput(fis, [80 200], 'Name', 'BloodPressure');
fis = addMF(fis, 'BloodPressure', 'trapmf', [80 80 100 120], 'Name', 'Normal');
fis = addMF(fis, 'BloodPressure', 'trapmf', [100 120 140 160], 'Name', 'Elevated');
fis = addMF(fis, 'BloodPressure', 'trapmf', [140 160 200 200], 'Name', 'High');

% Output Risk
fis = addOutput(fis, [0 1], 'Name', 'Risk');
fis = addMF(fis, 'Risk', 'trimf', [0 0 0.5], 'Name', 'Low');
fis = addMF(fis, 'Risk', 'trimf', [0.25 0.5 0.75], 'Name', 'Moderate');
fis = addMF(fis, 'Risk', 'trimf', [0.5 1 1], 'Name', 'High');

%% Define Rules
rules = ["If Age is Young and Cholesterol is Low and BloodPressure is Normal then Risk is Low", ...
         "If Age is MiddleAged and Cholesterol is Moderate and BloodPressure is Elevated then Risk is Moderate", ...
         "If Age is Old and Cholesterol is High and BloodPressure is High then Risk is High"];

fis = addRule(fis, rules);

%% Train Neuro-Fuzzy System (ANFIS)
% Prepare input and output for ANFIS
BPValues = cellfun(@(x) sscanf(x, '%d/%d'), BloodPressure, 'UniformOutput', false);
BPMatrix = cell2mat(BPValues');
BPMean = mean(BPMatrix, 2); % Mean Arterial Pressure

inputs = [Age Cholesterol BPMean];
anfis_data = [inputs, output];

% Split Data
cv = cvpartition(size(anfis_data, 1), 'HoldOut', 0.3);
idx = cv.test;
trainData = anfis_data(~idx, :);
testData = anfis_data(idx, :);

% Train FIS using ANFIS
[trainedFIS, trainError] = anfis(trainData, fis);

%% Predict with Trained FIS
anfisOutput = evalfis(trainedFIS, testData(:, 1:3));

%% ANN Integration
% Use FIS output as an additional feature for ANN
XTrain = [trainData(:, 1:3), evalfis(trainedFIS, trainData(:, 1:3))];
XTest = [testData(:, 1:3), anfisOutput];
YTrain = trainData(:, end);
YTest = testData(:, end);

% Standardize Data
meanX = mean(XTrain);
stdX = std(XTrain);
XTrain = (XTrain - meanX) ./ stdX;
XTest = (XTest - meanX) ./ stdX;

% Define and Train ANN
hiddenLayerSizes = 10;
Mdl = fitcnet(XTrain, YTrain, 'LayerSizes', hiddenLayerSizes);

% Make Predictions
YPred = predict(Mdl, XTest);

%% Evaluate Performance
accuracy = sum(round(YPred) == YTest) / length(YTest) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);
confusionchart(YTest,round(YPred));
