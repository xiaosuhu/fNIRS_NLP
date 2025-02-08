%% Find out which one has no stim
datadir = 'F:\MatlabBackUp\PROJECT_HH_ENGLISH\Pilot_Aanalysis_LLM_data_prep\rawdata_v2_deslxic_regular';
dataset = nirs.io.loadDirectory(datadir,{'Group', 'Subject'});

for i = 1:length(dataset)
   if isempty(dataset(i).stimulus)
       disp(i)
       disp(dataset(i).description)
   end
end