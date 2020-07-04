function Login(){
   var un=document.getElementById("id");
   var us=document.getElementById("password");
   var er=document.getElementById("error_box");
   er.innerHTML = "<br>";
   if(un.value.length<6||un.value.length>20){
      er.innerHTML="用户名必须在6-20个字符之间";
      return false;
   }else if((un.value.charCodeAt(0))>=48 && un.value.charCodeAt(0)<=57){
      er.innerHTML = "首字母不能为数字"
      return false;
   }else for(var i=0;i<un.value.length;i++){
       if((un.value.charCodeAt(i)<48)||(un.value.charCodeAt(i)>57)&&(un.value.charCodeAt(i)<97)&&(un.value.charCodeAt(i)>122)){
           er.innerHTML="用户名只能为数字和字母"
           return false;
       }
   }
   if(us.value.length<6||us.value.length>20){
   　　er.innerHTML="密码必须在6-20个字符之间";
   　　return false;
   }
 return true;
}

<input type="submit" class="btn btn-default" onclick="return Login()" value="登陆"></input>