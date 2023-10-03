

%%



Settle = datetime(2019,9,15);
Maturity = datetime(2023,9,15);
Rate = 0.035;
myRC = ratecurve('zero',Settle,Maturity,Rate,'Basis',12);

%%

VolRR = -0.0045;
VolBF = 0.0037;
RateF = 0.0210;
outPricer = finpricer("VannaVolga","DiscountCurve",myRC,"Model",BlackScholesModel,'SpotPrice',100,'DividendValue',RateF,'VolatilityRR',VolRR,'VolatilityBF',VolBF)

[Price, outPR] = price(outPricer,DoubleBarrierOpt,["all"])

%% 


DoubleBarrierOpt = fininstrument("DoubleBarrier",'Strike',100,'ExerciseDate',datetime(2020,8,15),'OptionType',"call",'ExerciseStyle',"European",'BarrierType',"DKO",'BarrierValue',[110 80],'Name',"doublebarrier_option")