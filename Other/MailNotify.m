function MailNotify(varargin)
%MAILNOTIFY Summary of this function goes here
%   Detailed explanation goes here

    try
        mail = getAdditionalParam( 'mail', varargin, 'wks.storage@gmail.com' ); % Origin email
        password = getAdditionalParam( 'password', varargin, 'WKstorage01' );
        host = getAdditionalParam( 'host', varargin, 'smtp.gmail.com' );
        sendto = getAdditionalParam( 'sendto', varargin, 'wks.storage@gmail.com' ); % Destination email
        time_now = getAdditionalParam( 'time_now', varargin, char(datetime('now','TimeZone','+07:00')) );
        Subject = getAdditionalParam( 'Subject', varargin, ['Finished running experiment at ' time_now] );
        Message = getAdditionalParam( 'Message', varargin, ['Your running experiments have finished at: ' time_now] );

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
    
    catch ME
        warning(['ERROR sending email ' ME.message]);
    end

end

