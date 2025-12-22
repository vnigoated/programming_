Create Table person(
id INT,
name Varchar(100),
city Varchar(100)
);

select * from person;


insert into person(id, name, city) 
values(102,'abc','pune'),
(103,'xyz','pune'),
(104,'def','mumbai');

update person
set city = 'mambai'
where id = 103


