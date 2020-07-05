$(document).ready(function(){
	console.log("Asdfdsa");
	$.ajax({
      url:"status",
      type:'post',
      dataType:'json',
      success:function(status){
      	if(status=='1'){
      		$("h1").html("Status: Waitconfirm");
      		$("p").html("Please be patient.The teacher is reviewing your application.");
      	}
      	else if(status=='2'){
      		$("h1").html("Status: Accepted");
      		$("p").html("Congratulations!You've been accepted!");
      	}
      	else{
      		$("h1").html("Status: Failed");
      		$("p").html("Sorry.Your application failed for the following reasons:");
      		$.ajax({
		      url:"status",
		      type:'post',
		      dataType:'json',
		      success:function(result){
		      	$(".container").append("<p>"+reasult.reason+"</p>");
		      }
      		});
      	}
      }
    });
});