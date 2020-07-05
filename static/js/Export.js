var form;var tot=0;
function search(){ 
    var search=$("#Search").val();
    var infor;var cnt=0;
    for(var i=0;i<search.length;++i)
    {
        if(search[i]=='_')
        {
            cnt=cnt+1;
        }
        else{
            infor[cnt]+=search[i];
        }
    }
    $.ajax({
            url : "lookup.html",
            type :'post',
            dataType:'json',
            success :function(fform){
                for(var i=0;i<fform.length;++i)
                {
                    if((infor[0]==''||infor[0]==fform[i].First_Name)&&(infor[1]==''||infor[1]==fform[i].Last_Name)&&(infor[2]==''||infor[2]==fform[i].Nationality)&&(infor[3]==''||infor[3]==fform[i].Year))
                    {
                        $(".table").append("<tr><td>"+fform[i].Name+"</td><td>"+fform[i].Nationality+"</td><td>"+fform[i].Sex"</td><td>"+fform[i].Name_Of_Institution+"</td><td>"+fform[i].Year+"</td><td><button type="button" class="btn btn-primary" style="text-algin:center">查看信息</button></td></tr>"); 
                        form[tot]=fform[i]
                        tot=tot+1; 
                    }
                    
                }
            }
        });
}
function expot(){
    let str = `First_Name,Last_Name,Chinese_Name,Sex,Nationality,Date_Of_Birth,Place_of_Birth,Mather_Tongue,Religion,Health_Condition,Name_Of_Institution,Highest_Education,Email,Phone_Number,Passport_Number,Date_Of_Expiry,Emergency_Name,Emergency_Relationship,Emergency_Phone,Emergency_email,Name_Of_Sponsor,Sponsor_Relationship,Sponsor_Phone,Sponsor_Email,Mail_Recipient,Mail_Phone,Mail_Address,Mail_City,Mail_Country,Mail_Postcode\n`;      
    //增加\t为了不让表格显示科学计数法或者其他格式      
    for(let i = 0 ; i < form.length ; i++ ){ 
           for(let item in form[i]){
                       str+=`${form[i][item] + '\t'},`;      
           }      
             str+='\n'; 
     }      //encodeURIComponent解决中文乱码      
    let uri = 'data:text/csv;charset=utf-8,\ufeff' + encodeURIComponent(str);      //通过创建a标签实现      
    var link = document.createElement("a");     
    link.href = uri;      //对下载的文件命名     
    link.download =  "json数据表.csv";      
    document.body.appendChild(link);    
    link.click();   
    document.body.removeChild(link); 
}
