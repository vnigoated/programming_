select * from bankdata
where 
salary between 40000 and 50000

select distinct dept from bankdata

select * from bankdata
order by fname desc

select * from bankdata where fname like '%a'

select * from bankdata where fname like '%a%'

select * from bankdata where dept like '__'

select count(emp_id) from bankdata;

select avg(salary) from bankdata;
select min(salary) from bankdata;

select dept from bankdata group by dept;

select dept  , count(emp_id) from bankdata group by dept;

select concat('hello','world')

select concat_ws(fname,' ',lname) as fullname  from bankdata

select replace ('abcxyz','abc','pqr')

select replace(dept,'IT','tech')from bankdata










