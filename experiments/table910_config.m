function cfg = table910_config()
%TABLE910_CONFIG Dataset metadata and V4 reproduction settings.

    cfg.datasets = {'20NEWS','CANCER','CITESEER','FOOTBALL','ORL', ...
                    'PROTEIN','REUTERS','SEG','SPAM','UMIST'};
    cfg.samples = [18846,190,3312,115,400,17766,8293,2310,4601,575];
    cfg.classes = [20,14,6,12,40,3,65,7,2,20];
    cfg.seeds = 1:30;
    cfg.kmeans_replicates = 10;
    cfg.kmeans_max_iter = 300;
end
