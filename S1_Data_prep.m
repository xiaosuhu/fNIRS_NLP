%% Create the hb object

datafolder='F:\MatlabBackUp\PROJECT_HH_ENGLISH\Pilot_Aanalysis_LLM_data_prep\story_data_v2\WD';
% datafolder='/Volumes/Franknano/MatlabBackUp/PROJECT_DyslexiaR01/HHData_Cleaned';
raw = nirs.io.loadDirectory(datafolder,{'Subject'});

disp('Running data resample...')
resample=nirs.modules.Resample();
resample.Fs=5;
downraw=resample.run(raw);

disp('Converting Optical Density...')
odconv=nirs.modules.OpticalDensity();
od=odconv.run(downraw);

disp('Applying  Modified Beer Lambert Law...')
mbll=nirs.modules.BeerLambertLaw();
hb=mbll.run(od);

% for i = 1:length(hb)
%     if length(hb(i).time) == length(hb(i).data)
%         disp(1)
%     else
%         disp(0)
%     end
% end

disp('Trimming .nirs files...')
trim=nirs.modules.TrimBaseline();
trim.preBaseline=0;
trim.postBaseline=30;
hb_trim=trim.run(hb);

%% Save the hb data with fixed length 4180 to ignore the 0 section

for i = 1:length(hb)
    try
        hbodata = hb_trim(i).data(1:4180,1:2:40);
    catch
        disp(strcat(hb_trim(i).description,' not fit...'));
    end
    [folder, baseFileName, ext] = fileparts(hb(i).description);
    ext='.mat';
    fileNameWithExt = [baseFileName, ext];
    save(strcat('./hbodata_v2_WD/',fileNameWithExt), 'hbodata');
    disp(strcat(num2str(i),'...'));
end
