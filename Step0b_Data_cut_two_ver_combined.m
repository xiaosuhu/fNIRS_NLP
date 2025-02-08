%% Combine the two versions of data cut to split the story and question part

datadir = 'F:\MatlabBackUp\PROJECT_HH_ENGLISH\Pilot_Aanalysis_LLM_data_prep\rawdata_v2_deslxic_regular\WD';
dataset = nirs.io.loadDirectory(datadir,{'Subject'});

outputdir_story = 'F:\MatlabBackUp\PROJECT_HH_ENGLISH\Pilot_Aanalysis_LLM_data_prep\story_data\WD';
% outputdir_story = 'F:\MatlabBackUp\PROJECT_HH_ENGLISH\Pilot_Aanalysis_LLM_data_prep\story_WD';

% outputdir_qa = 'F:\MatlabBackUp\PROJECT_HH_ENGLISH\Data_cut_Data\HHdata_AI4094_cut\qa';


for i = 8:length(dataset)
    load(dataset(i).description, '-mat');
    
    [~,filename,fileext] = fileparts(dataset(i).description);
    id = strsplit(filename, '_');
    id = id{1};
    
    try
        % Get the stim mark pairs for the 12 question version
        stim1_diff=diff(s(:, 1));
        stim2_diff=diff(s(:, 2));
        stim3_diff=diff(s(:, 3));
        
        stim1_mark = find(stim1_diff==1);
        stim2_mark = find(stim2_diff==1);
        stim3_mark = find(stim3_diff==1);
        
        story_pairs =[stim1_mark(1) stim3_mark(1)
            stim1_mark(2) stim2_mark(1)
            stim1_mark(3) stim3_mark(2)
            stim1_mark(4) stim2_mark(2)
            stim1_mark(5) stim2_mark(3)
            stim1_mark(6) stim3_mark(3)
            stim1_mark(7) stim3_mark(4)
            stim1_mark(8) stim3_mark(5)
            stim1_mark(9) stim2_mark(4)
            stim1_mark(10) stim2_mark(5)
            stim1_mark(11) stim2_mark(6)
            stim1_mark(12) stim3_mark(6)
            ];
        qa_pairs = [stim3_mark(1) stim1_mark(2)
            stim2_mark(1) stim1_mark(3)
            stim3_mark(2) stim1_mark(4)
            stim2_mark(2) stim1_mark(5)
            stim2_mark(3) stim1_mark(6)
            stim3_mark(3) stim1_mark(7)
            stim3_mark(4) stim1_mark(8)
            stim3_mark(5) stim1_mark(9)
            stim2_mark(4) stim1_mark(10)
            stim2_mark(5) stim1_mark(11)
            stim2_mark(6) stim1_mark(12)
            stim3_mark(6) length(s)
            ];
        
        % Cut the data story part
        story.d = [];
        story.t = [];
        story.s = [];
        story.aux = [];
        story.SD = SD;
        for j = 1:length(story_pairs)
            story.d = [story.d;d(story_pairs(j,1):story_pairs(j, 2), :)];
            story.t = [story.t;t(story_pairs(j,1):story_pairs(j, 2))];
            story.s = [story.s;s(story_pairs(j,1):story_pairs(j, 2),1)];
            story.aux = [story.aux ;aux(story_pairs(j,1):story_pairs(j, 2), :, :)];
        end
        
        
        mkdir(strcat(outputdir_story, '\', id))
        cd(strcat(outputdir_story, '\', id))
        save(strcat('story_',id,'.nirs'),'-struct','story')
        % Cut the data qa part
        %     qa.d = []; % Not needed as it will save every loop
        %     qa.t = [];
        %     qa.s = [];
        %     qa.aux = [];
        %     mkdir(strcat(outputdir_qa, '\', id))
        %     qa.SD = SD;
        %     for j = 1:length(qa_pairs)
        %         qa.d = d(qa_pairs(j,1):qa_pairs(j, 2), :);
        %         qa.t = t(qa_pairs(j,1):qa_pairs(j, 2));
        %         qa.s = s(qa_pairs(j,1):qa_pairs(j, 2),1);
        %         qa.aux = aux(qa_pairs(j,1):qa_pairs(j, 2), :, :);
        %
        %         % Save
        %         mkdir(strcat(outputdir_qa, '\', id, '\q',num2str(j)));
        %         cd(strcat(outputdir_qa, '\', id, '\q',num2str(j)))
        %
        %         save(strcat(id,'_q_',num2str(j),'.nirs'),'-struct','qa')
        %
        %     end
    catch
        % Get the stim mark pairs
        stim1_diff=diff(s(:, 1));
        stim2_diff=diff(s(:, 2));
        try
            stim3_diff=diff(s(:, 4));
        catch
            stim3_diff=diff(s(:, 3));
        end
        
        
        stim1_mark = find(stim1_diff==1);
        stim2_mark = find(stim2_diff==1);
        stim3_mark = find(stim3_diff==1);
        
        story_pairs =[stim1_mark(1) stim2_mark(1)
            stim3_mark(1) stim2_mark(2)
            stim3_mark(2) stim2_mark(3)
            stim3_mark(3) stim2_mark(4)
            stim3_mark(4) length(s)
            ];
        %     qa_pairs = [stim3_mark(1) stim1_mark(2)
        %                   stim2_mark(1) stim1_mark(3)
        %                   stim3_mark(2) stim1_mark(4)
        %                   stim2_mark(2) stim1_mark(5)
        %                   stim2_mark(3) stim1_mark(6)
        %                   stim3_mark(3) stim1_mark(7)
        %                   stim3_mark(4) stim1_mark(8)
        %                   stim3_mark(5) stim1_mark(9)
        %                   stim2_mark(4) stim1_mark(10)
        %                   stim2_mark(5) stim1_mark(11)
        %                   stim2_mark(6) stim1_mark(12)
        %                   stim3_mark(6) length(s)
        %                   ];
        
        Fs = 1/mean(diff(t));
        % Cut the data story part
        story.d = d(story_pairs(1,1)-30:story_pairs(1,1),:);
        % story.t = [];
        story.s = s(story_pairs(1,1)-30:story_pairs(1,1),1:3);
        story.aux = [];
        story.SD = SD;
        for j = 1:length(story_pairs)
            story.d = [story.d;d(story_pairs(j,1):story_pairs(j, 2), :)];
            % story.t = [story.t;t(story_pairs(j,1):story_pairs(j, 2))];
            story.s = [story.s;s(story_pairs(j,1):story_pairs(j, 2),1:3)];
            % story.aux = [story.aux ;aux(story_pairs(j,1):story_pairs(j, 2), :, :)];
        end
        
        % align the time and make it continously
        story.t = (1/Fs : 1/Fs : size(story.d,1)/Fs)';
        % Make the s 3 columns to match the original
        story.s(2,2)=1;
        story.s(3,3)=1;
        
        mkdir(strcat(outputdir_story, '/', id))
        cd(strcat(outputdir_story, '/', id))
        save(strcat('story_',id,'.nirs'),'-struct','story')
        % Cut the data qa part
        %     qa.d = []; % Not needed as it will save every loop
        %     qa.t = [];
        %     qa.s = [];
        %     qa.aux = [];
        %     mkdir(strcat(outputdir_qa, '\', id))
        %     qa.SD = SD;
        %     for j = 1:length(qa_pairs)
        %         qa.d = d(qa_pairs(j,1):qa_pairs(j, 2), :);
        %         qa.t = t(qa_pairs(j,1):qa_pairs(j, 2));
        %         qa.s = s(qa_pairs(j,1):qa_pairs(j, 2),1);
        %         qa.aux = aux(qa_pairs(j,1):qa_pairs(j, 2), :, :);
        %
        %         % Save
        %         mkdir(strcat(outputdir_qa, '\', id, '\q',num2str(j)));
        %         cd(strcat(outputdir_qa, '\', id, '\q',num2str(j)))
        %
        %         save(strcat(id,'_q_',num2str(j),'.nirs'),'-struct','qa')
        %
        %     end
        
    end
    
    disp(strcat(id, ' Finished...'));
end