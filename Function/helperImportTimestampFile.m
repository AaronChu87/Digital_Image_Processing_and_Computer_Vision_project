function timestamp = helperImportTimestampFile(filename)

% Input handling
dataLines = [4, Inf];

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = " ";

% Specify column names and types
opts.VariableNames = ["VarName1", "Var2"];
opts.SelectedVariableNames = "VarName1";
opts.VariableTypes = ["double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";

% Specify variable properties
opts = setvaropts(opts, "Var2", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var2", "EmptyFieldRule", "auto");

% Import the data
data = readtable(filename, opts);

% Convert to output type
timestamp = table2array(data);
end