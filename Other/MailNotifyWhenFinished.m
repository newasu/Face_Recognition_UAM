
% parameters
mail = 'wks.storage@gmail.com'; % my gmail address
password = 'WKstorage01';  % my gmail password 
host = 'smtp.gmail.com';
sendto = 'wks.storage@gmail.com';
time_now = char(datetime('now','TimeZone','+07:00'));
Subject = ['Experiment finished ' time_now];
Message = ['Your running experiment was finish at: ' time_now];

% preferences
setpref('Internet','SMTP_Server', host);
setpref('Internet','E_mail',mail);
setpref('Internet','SMTP_Username',mail);
setpref('Internet','SMTP_Password',password);
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');

% execute
sendmail(sendto,Subject,Message)

