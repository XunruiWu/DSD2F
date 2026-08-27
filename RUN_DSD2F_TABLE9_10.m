% RUN_DSD2F_TABLE9_10
% One-command reproduction of the DSD2F columns of Tables 9 and 10.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'));
addpath(fullfile(root,'utils'));
addpath(fullfile(root,'experiments'));
summary = run_dsd2f_table910(); %#ok<NASGU>
