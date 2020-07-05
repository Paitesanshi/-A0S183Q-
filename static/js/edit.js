function search(){
	var Name=$("#name");
	var Nationality=$("#nationality");
	var Year=$("#year");
	$.ajax({
			url : "edit.html",
			type :'post',
			data :{Name:Name,Nationality=Nationality,Year=Year}
			dataType:'json',
			success :function(fform){
				for(var i=0;i<fform.length;++i)
				{
					$(".table").append("<tr><input type="checkbox"> <td>"+fform[i].Name+"</td><td>"+fform[i].Sex+"</td><td>"+fform[i].Nationality+"</td></tr>");
				}
			}
		});
	$("tr").click(function(){
		var x=$(this).index();
		var user_email=fform[x-1].email;
	$.ajax({
            url: "test_post",
            data: { email:user_email},
            dataType: "json",
            type:"POST",
            success: function (result) {
                window.location.href="checkuser.html";
            }
        })
    });
    $("tr").hover(
    	function(){
       		$(this).css("background-color","white");
    	},
    	function(){
       		$(this).css("background-color","");
    	}
    );
}
function exportExcel(){
	var x=$("input[type=checkbox]:checked");
	var user_email;
	for(var i=0;i<x.length;++i){
		user_email[i]=fform[x[i].index()-1].email;
	}
	$.ajax({
			url : "edit.html",
			type :'post',
			data :{user_email:user_email}
			dataType:'json',
			success :function(fform){
			}
		});
}