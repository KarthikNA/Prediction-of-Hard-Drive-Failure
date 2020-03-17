.echo on

--
-- Create a table that has the number of drive days for each
-- model, which is simply the number of rows in drive_stats
-- for that model.
--
CREATE TABLE drive_days AS 
    SELECT model, count(*) AS drive_days 
    FROM drive_stats 
    GROUP BY model;

--
-- Create a table that has the number of failures for each model.
--
CREATE TABLE failures AS
    SELECT model, count(*) AS failures
    FROM drive_stats
    WHERE failure = 1
    GROUP BY model;

--
-- Create a table that has the number of drives for each model
-- as of June 30, 2019.
--
CREATE TABLE model_count AS
    SELECT model, count(*) AS count
    FROM drive_stats
    WHERE date = '2019-12-31'
    GROUP BY model;

--
-- Join the tables together and compute the annual failure rate.
-- "drive years" is computed by dividing the number of drive days
-- by 365, and then the annual failure rate is simply the number
-- of failures divided by the number of drive years.  The result
-- is multiplied by 100 to get a percentage.
--
CREATE TABLE failure_rates AS
    SELECT drive_days.model AS model,
           drive_days.drive_days AS drive_days,
           failures.failures AS failures, 
           100.0 * (1.0 * failures) / (drive_days / 365.0) AS annual_failure_rate
    FROM drive_days, failures, model_count
    WHERE drive_days.model = failures.model
      AND model_count.model = failures.model
    ORDER BY model;

.echo off
