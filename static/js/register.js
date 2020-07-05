$('.form').find('input, textarea').on('keyup blur focus', function (e) {
  
  var $this = $(this),
      label = $this.prev('label');

	  if (e.type === 'keyup') {
			if ($this.val() === '') {
          label.removeClass('active highlight');
        } else {
          label.addClass('active highlight');
        }
    } else if (e.type === 'blur') {
    	if( $this.val() === '' ) {
    		label.removeClass('active highlight'); 
			} else {
		    label.removeClass('highlight');   
			}   
    } else if (e.type === 'focus') {
      
      if( $this.val() === '' ) {
    		label.removeClass('highlight'); 
			} 
      else if( $this.val() !== '' ) {
		    label.addClass('highlight');
			}
    }

});

$('.tab a').on('click', function (e) {
  
  e.preventDefault();
  
  $(this).parent().addClass('active');
  $(this).parent().siblings().removeClass('active');
  
  target = $(this).attr('href');

  $('.tab-content > div').not(target).hide();
  
  $(target).fadeIn(600);
  
});
function check(){
  var x=document.getElementById("userName");
  x.setCustomValidity("");
  if(x.checkValidity()==false){
    x.setCustomValidity("Your e-mail address is wrong!");
    return;
  }
  var vv=document.getElementById("userPsd");
  vv.setCustomValidity("");
  if(vv.value.length<6)
  {
    vv.setCustomValidity('Your password must include at least six letters or Numbers');
    return;
  }
  var cv=document.getElementById("conPsd");
  cv.setCustomValidity("");
  if(cv.value!=vv.value)
  {
    cv.setCustomValidity('The two passwords do not match');
    return;
  }
  var email=$("userName").val();
  var pass=$("userPsd").val();
  var user={
  mail:email,password:pass
};  
    document.write("");
    $.ajax({
      url:"register",
      type:'post',
      data:JSON.stringify(user),
      contentType:'application/json; charset=utf-8',
      dataType:'json',
      success:function(data){
      }
    });
    window.location.href="information.html";
}
function check2(){
  var x=document.getElementById("userName");
  var vv=document.getElementById("userPsd");
  vv.setCustomValidity("");
  x.setCustomValidity("");
  if(x.value=='admin'&&vv.value=='')
  {
     window.location.href="administrator.html";
     return;
  }
  if(x.checkValidity()==false){
    x.setCustomValidity("Your e-mail address is wrong!");
    return;
  }
  if(vv.value.length<6)
  {
    vv.setCustomValidity('Your password must include at least six letters or Numbers');
    return;
  }
  var email=$("userName").val();
  var pass=$("userPsd").val();
  var user={
  mail:email,password:pass
};  
    document.write("");
    $.ajax({
      url:"register",
      type:'post',
      data:JSON.stringify(user),
      contentType:'application/json; charset=utf-8',
      dataType:'json',
      success:function(result){
        if(result==1)
        {
          vv.setCustomValidity('Your password is wrong');
        }
        else{
          window.location.href="information.html";
        }
      }
    });
}
// var email=$("userName").val();
// var pass=$("userPsd").val();
// var user={
//   mail:email,password:pass
// };  
// $("button").submit(function(){
// document.write("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");console.log("aaaaa");
//     $.ajax({
//       url:"register",
//       type:'post',
//       data:JSON.stringify(user),
//       contentType:'application/json; charset=utf-8',
//       dataType:'json',
//       success:function(data){
//         window.location.href="information.html";
//       }
//     });
    
//     window.location.href="information.html";
// });
var slideShowContainer = document.getElementById("slideShowContainer");
var pic = document.getElementById("picUl").getElementsByTagName("li");
var dot = document.getElementById("dotUl").getElementsByTagName("li");
var title = document.getElementById("titleDiv").getElementsByTagName("span");
var index = 0;
var timer = null;
/*定义图片切换函数*/
function changePic (curIndex) {
    for(var i = 0;i < pic.length;++i){
        pic[i].style.display = "none";
        dot[i].className = "unselected";
        title[i].className = "hide"
    }
    pic[curIndex].style.display = "block";
    dot[curIndex].className = "selected";
    title[curIndex].className = "show";
}
/*index超出图片总量时归零*/
function autoPlay(){
    if(+index >= pic.length){
        index = 0;
    }
    changePic(index);
    index++;
}
/*定义并调用自动播放函数*/
timer = setInterval(autoPlay,1500);
/*鼠标划过整个容器时停止自动播放*/
slideShowContainer.onmouseover = function(){
    clearInterval(timer);
}
/*鼠标离开整个容器时继续播放下一张*/
slideShowContainer.onmouseout = function(){
    timer = setInterval(autoPlay,1500);
}
/*遍历所有数字导航实现划过切换至对应的图片*/
for(var i = 0;i < dot.length;i++){
    dot[i].onmouseover = function(){
        clearInterval(timer);
        index = this.innerText-1;
        changePic(index)
    }
}