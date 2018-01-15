
feat = {'mfeat-fou','mfeat-fac','mfeat-kar','mfeat-pix','mfeat-zer','mfeat-mor'};

acc = zeros(1,6);
for f=1:6
    fdir = feat{f};
    feature = ['I:\mydoc\matlab\HandWritten\mfeat\mfeat\' fdir  ];
    
    mfeat_fou = load (feature);

    label = [10*ones(1,200) ones(1,200) 2*ones(1,200) 3*ones(1,200) 4*ones(1,200) 5*ones(1,200) 6*ones(1,200) 7*ones(1,200) 8*ones(1,200) 9*ones(1,200)]';
 
    hw_train_label=label([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960]);%每类取40个数据作为训练，共120个训练数据
    hw_train_data=mfeat_fou([1:160 201:360 401:560 601:760 801:960 1001:1160 1201:1360 1401:1560 1601:1760 1801:1960],:);
    hw_test_label=label([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000]);
    hw_test_data=mfeat_fou([161:200 361:400 561:600 761:800 961:1000 1161:1200 1361:1400 1561:1600 1761:1800 1961:2000],:);
    
    model=svmtrain(hw_train_label,hw_train_data);
    [hw_predict_label,hw_accuracy,e]=svmpredict(hw_test_label,hw_test_data,model,'-q');

feat{f}
hw_accuracy
acc(f) = hw_accuracy(1);
acc
end