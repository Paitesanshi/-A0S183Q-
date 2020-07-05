$(document).ready(function(){
	$.ajax({
                url: "information",
                dataType: "json",
                type:"POST",
                success: function (fform) {
                	for(var i=0;i<fform.length;++i){
                		$(".list").append("<tr><td>"+fform[i].Name+"</td><td>"+fform[i].Sex+"</td><td>"+fform[i].Nationality+"</td></tr>");
                	}
                    $("tr").click(function(){
                    var x=$(this).index();
                    var user_email=fform[x-1].email;
                    $.ajax({
                        url: "test_post",
                        data: { email:user_email},
                        dataType: "json",
                        type:"POST",
                        success: function (result) {
                            window.location.href="information.html";
                        }
                          })
                    });
                }
            });
	
    $("tr").hover(
    	function(){
       		$(this).css("background-color","white");
    	},
    	function(){
       		$(this).css("background-color","");
    	}
    );
});