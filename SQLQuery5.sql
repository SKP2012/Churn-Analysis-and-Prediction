-- Creating Views just to verify/check if the possible table---

create view vw_ChurnData as	
	select * from prod_Churn where Customer_Status in ('Churned','Stayed');

create view vw_JoinData as	
	select * from prod_Churn where Customer_Status = 'Joined';

	sp_helptext vw_ChurnData

	--- Check created view: select * from  vw_ChurnData ---