test = squeeze(mean(brain_score, 1))';

roi1=mean(test(:,[3]), 2);
roi2=mean(test(:,[9 10]), 2);

figure
title('IFG-vs-STG')
plot(roi2-roi2(1));
hold on
plot(roi1-roi1(1));
legend('STG','IFG')
grid on
