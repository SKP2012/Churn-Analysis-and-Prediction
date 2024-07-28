--Basic Information Analysis--
-- Data Exploration - Check Distinct Values--

select Gender, Count(Gender) as TotalCount,
Count(Gender)*100.00 /(select Count(*) from stg_Churn) as Percentage
from stg_Churn
Group by Gender


select Contract, Count(Contract) as TotalCount,
Count(Contract)*100.00 /(select Count(*) from stg_Churn) as Percentage
from stg_Churn
Group by Contract

select Customer_Status, Count(Customer_Status) as TotalCount, Sum(Total_Revenue) as TotalRev,
Sum(Total_Revenue)*100 /(select sum(Total_Revenue) from stg_Churn) as RevPercentage
from stg_Churn
Group by Customer_Status

select State, Count(State) as TotalCount,
Count(State)*100.0 /(select Count(*) from stg_Churn) as Percentage
from stg_Churn
Group by State
Order by Percentage desc

select Gender, count(*) as Totalc
from stg_Churn
group by Gender

select distinct(Internet_Type)
from stg_Churn