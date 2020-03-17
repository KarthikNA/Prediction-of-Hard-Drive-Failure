--
-- Import each of the daily drive stats files for Q4 2019 ONLY
--

.mode csv
.echo on
.import ./2019/2019-10-01.csv drive_stats
.import ./2019/2019-10-02.csv drive_stats
.import ./2019/2019-10-03.csv drive_stats
.import ./2019/2019-10-04.csv drive_stats
.import ./2019/2019-10-05.csv drive_stats
.import ./2019/2019-10-06.csv drive_stats
.import ./2019/2019-10-07.csv drive_stats
.import ./2019/2019-10-08.csv drive_stats
.import ./2019/2019-10-09.csv drive_stats
.import ./2019/2019-10-10.csv drive_stats
.import ./2019/2019-10-11.csv drive_stats
.import ./2019/2019-10-12.csv drive_stats
.import ./2019/2019-10-13.csv drive_stats
.import ./2019/2019-10-14.csv drive_stats
.import ./2019/2019-10-15.csv drive_stats
.import ./2019/2019-10-16.csv drive_stats
.import ./2019/2019-10-17.csv drive_stats
.import ./2019/2019-10-18.csv drive_stats
.import ./2019/2019-10-19.csv drive_stats
.import ./2019/2019-10-20.csv drive_stats
.import ./2019/2019-10-21.csv drive_stats
.import ./2019/2019-10-22.csv drive_stats
.import ./2019/2019-10-23.csv drive_stats
.import ./2019/2019-10-24.csv drive_stats
.import ./2019/2019-10-25.csv drive_stats
.import ./2019/2019-10-26.csv drive_stats
.import ./2019/2019-10-27.csv drive_stats
.import ./2019/2019-10-28.csv drive_stats
.import ./2019/2019-10-29.csv drive_stats
.import ./2019/2019-10-30.csv drive_stats
.import ./2019/2019-10-31.csv drive_stats
.import ./2019/2019-11-01.csv drive_stats
.import ./2019/2019-11-02.csv drive_stats
.import ./2019/2019-11-03.csv drive_stats
.import ./2019/2019-11-04.csv drive_stats
.import ./2019/2019-11-05.csv drive_stats
.import ./2019/2019-11-06.csv drive_stats
.import ./2019/2019-11-07.csv drive_stats
.import ./2019/2019-11-08.csv drive_stats
.import ./2019/2019-11-09.csv drive_stats
.import ./2019/2019-11-10.csv drive_stats
.import ./2019/2019-11-11.csv drive_stats
.import ./2019/2019-11-12.csv drive_stats
.import ./2019/2019-11-13.csv drive_stats
.import ./2019/2019-11-14.csv drive_stats
.import ./2019/2019-11-15.csv drive_stats
.import ./2019/2019-11-16.csv drive_stats
.import ./2019/2019-11-17.csv drive_stats
.import ./2019/2019-11-18.csv drive_stats
.import ./2019/2019-11-19.csv drive_stats
.import ./2019/2019-11-20.csv drive_stats
.import ./2019/2019-11-21.csv drive_stats
.import ./2019/2019-11-22.csv drive_stats
.import ./2019/2019-11-23.csv drive_stats
.import ./2019/2019-11-24.csv drive_stats
.import ./2019/2019-11-25.csv drive_stats
.import ./2019/2019-11-26.csv drive_stats
.import ./2019/2019-11-27.csv drive_stats
.import ./2019/2019-11-28.csv drive_stats
.import ./2019/2019-11-29.csv drive_stats
.import ./2019/2019-11-30.csv drive_stats
.import ./2019/2019-12-01.csv drive_stats
.import ./2019/2019-12-02.csv drive_stats
.import ./2019/2019-12-03.csv drive_stats
.import ./2019/2019-12-04.csv drive_stats
.import ./2019/2019-12-05.csv drive_stats
.import ./2019/2019-12-06.csv drive_stats
.import ./2019/2019-12-07.csv drive_stats
.import ./2019/2019-12-08.csv drive_stats
.import ./2019/2019-12-09.csv drive_stats
.import ./2019/2019-12-10.csv drive_stats
.import ./2019/2019-12-11.csv drive_stats
.import ./2019/2019-12-12.csv drive_stats
.import ./2019/2019-12-13.csv drive_stats
.import ./2019/2019-12-14.csv drive_stats
.import ./2019/2019-12-15.csv drive_stats
.import ./2019/2019-12-16.csv drive_stats
.import ./2019/2019-12-17.csv drive_stats
.import ./2019/2019-12-18.csv drive_stats
.import ./2019/2019-12-19.csv drive_stats
.import ./2019/2019-12-20.csv drive_stats
.import ./2019/2019-12-21.csv drive_stats
.import ./2019/2019-12-22.csv drive_stats
.import ./2019/2019-12-23.csv drive_stats
.import ./2019/2019-12-24.csv drive_stats
.import ./2019/2019-12-25.csv drive_stats
.import ./2019/2019-12-26.csv drive_stats
.import ./2019/2019-12-27.csv drive_stats
.import ./2019/2019-12-28.csv drive_stats
.import ./2019/2019-12-29.csv drive_stats
.import ./2019/2019-12-30.csv drive_stats
.import ./2019/2019-12-31.csv drive_stats
.echo off
.mode list

--
-- The drive stats files each have a header row that labels the
-- columns.  sqlite doesn't understand this when importing from
-- CSV, so they header rows land in the table.  This removes them.
--
delete from drive_stats where model = 'model';
